import torch
from torch.optim import Optimizer
from typing import Callable, Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.onpolicy_grpo.utils import tokenize_prompt_and_output, get_reponse_log_probs

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str]
) -> tuple[torch.Tensor, dict[str, float]]:
    reward = []
    format_reward = []
    for i in range(len(rollout_responses)):
        rewards = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        reward.append(rewards["reward"])
        format_reward.append(rewards["format_reward"])


    return (
        torch.tensor(reward),
        {
            "mean_total_reward": sum(reward) / len(reward),
            "mean_format_reward": sum(format_reward) / len(reward)
        }
    )

def compute_group_normalized_reward(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["none", "mean"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["none", "mean", "std"] = "std"
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if baseline !="mean" or advantage_normalizer != "std":
        raise NotImplementedError
    # reshape[quention_size, group_size], must use returns
    rewards = raw_rewards.reshape(-1, group_size)

    reward_mean = torch.mean(rewards, dim=-1, keepdim=True)
    reward_std = torch.std(rewards, dim=-1, keepdim=True)

    normalized_reward = None
    if baseline == "mean" and advantage_normalizer == "std":
        normalized_reward = (rewards - reward_mean) / (reward_std + advantage_eps)

    return (
        normalized_reward.flatten(),
        {
            "reward_mean": reward_mean,
            "reward_std": reward_std
        }
    )


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "grpo", "gspo", "noclip"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError
    # raw_rewards_or_advatages shape[batch_size]
    # policy_log_probs shape [batch_size, seq_len]
    raw_rewards_or_advantages = raw_rewards_or_advantages.reshape(-1, 1)
    gradient_loss = - raw_rewards_or_advantages * policy_log_probs

    return (
        gradient_loss,
        {}
    )


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["constant", "sequence"] = "sequence",
    normalization_constant: int | None =None
) -> torch.Tensor:
    mask_per_token_policy_gradient_loss = per_token_policy_gradient_loss.masked_fill(~mask, 0)
    # only average on the valid tokens, can not use mean
    # and must first average on seq , then average on batch
    sequence_loss = mask_per_token_policy_gradient_loss.sum(dim=-1) / mask.sum(dim=-1)
    return sequence_loss.mean()

def grpo_train_step(
     model: PreTrainedModel,
     tokenizer: PreTrainedTokenizer,
     optimizer: Optimizer,
     gradient_accumulation_steps: int,
     max_grad_norm: float | None,
     reward_fn: Callable[[str, str], dict[str, float]],
     repeated_prompts: list[str],
     rollout_responses: list[str],
     repeated_ground_truths: list[str],
     group_size: int,
     # Reward normalization
     baseline: Literal["mean", "none"] = "mean",
     advantage_eps: float = 1e-6,
     advantage_normalizer: Literal["std", "none", "mean"] = "std",
     # Importance reweighting and clipping
     importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
     old_log_probs: torch.Tensor | None = None,
     cliprange: float | None = None,
     # Loss normalization
     loss_normalization: Literal["sequence", "constant"] = "sequence",
     normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    device = next(model.parameters()).device
    # 1.tokenize the prompt and output
    tokenizer_output = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenizer_output["input_ids"].to(device)
    labels = tokenizer_output["labels"].to(device)
    response_mask = tokenizer_output["response_mask"].to(device)

    # 2. compute rewards
    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    normalized_reward = compute_group_normalized_reward(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)[0].to(device)

    batch_loss = 0
    entropy_sum = 0.0
    token_count = 0
    # 3.gradient accumulation
    assert len(input_ids) % gradient_accumulation_steps == 0
    microbatch_size = len(input_ids) // gradient_accumulation_steps
    for i in range(0, len(input_ids), microbatch_size):
        inputs_microbatch = input_ids[i: i + microbatch_size, ...]
        labels_microbatch = labels[i: i + microbatch_size, ...]
        mask_microbatch = response_mask[i: i + microbatch_size, ...]
        reward_microbatch = normalized_reward[i: i + microbatch_size, ...]
        # calculate log_probs
        log_prob_output = get_reponse_log_probs(model, inputs_microbatch, labels_microbatch, return_token_entropy=True)
        log_probs = log_prob_output["log_probs"]
        entropy_sum += log_prob_output["token_entropy"].detach().masked_fill(~mask_microbatch, 0).sum().item()
        token_count += mask_microbatch.sum().item()
        # compute policy gradient loss
        per_token_policy_gradient_loss = compute_policy_gradient_loss(reward_microbatch, log_probs, importance_reweighting_method, old_log_probs, cliprange, mask_microbatch)[0]
        # aggregate loss
        microbatch_loss = microbatch_size / len(input_ids) * aggregate_loss_across_microbatch(per_token_policy_gradient_loss, mask_microbatch, loss_normalization, normalization_constant)
        batch_loss += microbatch_loss.item()
        # backward and accumulate gradient
        microbatch_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad()

    return (
        torch.tensor(batch_loss),
        {**reward_metadata, "grad_norm": grad_norm.item(), "token_entropy": entropy_sum / max(token_count, 1)}
    )
