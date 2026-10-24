"""Compare pruning against a full-batch objective with existing Adam moments."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cs336_alignment.onpolicy_grpo.utils import tokenize_prompt_and_output
from scripts.train_grpo_kaggle import CPUAdamW, VARIANTS, train_batch


def score(response, truth):
    return dict(reward=float(response == "world"), format_reward=1.)


def full_loss(model, tokenizer, prompts, responses, variant):
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    logits = model(batch["input_ids"]).logits
    logp = logits.log_softmax(-1).gather(-1, batch["labels"].unsqueeze(-1)).squeeze(-1)
    rewards = torch.tensor([float(r == "world") for r in responses]).reshape(-1, 2)
    advantage = rewards if variant == "rft" else rewards - rewards.mean(-1, keepdim=True)
    if variant in ("grpo", "grpo_constant"):
        advantage = advantage / (rewards.std(-1, keepdim=True) + 1e-6)
    elif variant == "maxrl":
        advantage = advantage / (rewards.mean(-1, keepdim=True) + 1e-6)
    sums = logp.masked_fill(~batch["response_mask"], 0).sum(-1)
    if variant == "grpo":
        return -(advantage.flatten() * sums / batch["response_mask"].sum(-1)).mean()
    return -(advantage.flatten() * sums).sum() / (len(prompts) * 8)


def warm_moments(model, reference):
    actual = CPUAdamW(model.parameters(), lr=1e-3)
    expected = torch.optim.AdamW(reference.parameters(), lr=1e-3, betas=(.9, .95),
                                 weight_decay=0., foreach=False)
    rng = torch.Generator().manual_seed(52)
    for _ in range(3):
        for p, q in zip(model.parameters(), reference.parameters()):
            p.grad = torch.randn(p.shape, generator=rng)
            q.grad = p.grad.clone()
        actual.step()
        expected.step()
    actual.zero_grad(set_to_none=True)
    expected.zero_grad(set_to_none=True)
    return actual, expected


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_pruning_gradients_and_adam_update_with_history(tiny_train_model, tokenizer, variant, microbatch):
    reference = deepcopy(tiny_train_model)
    actual, expected = warm_moments(tiny_train_model, reference)
    prompts = ["Hello"] * 8
    # Mixed groups, all-wrong group, all-correct group, unequal lengths.
    responses = ["world", "a test", "test", "a test", "world", "world", "a test", "world"]
    args = SimpleNamespace(group_size=2, max_new_tokens=8, microbatch_size=microbatch,
                           gradient_accumulation_steps=2, max_grad_norm=.01)
    captured = []
    actual.register_step_pre_hook(lambda opt, a, kw:
        captured.append([p.grad.clone() for p in tiny_train_model.parameters()]))
    for _ in range(2):
        train_batch(tiny_train_model, tokenizer, actual, torch.amp.GradScaler("cuda", enabled=False),
                    prompts, responses, ["42"] * 8, score, args, VARIANTS[variant])
        expected.zero_grad(set_to_none=True)
        full_loss(reference, tokenizer, prompts, responses, variant).backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .01)
        assert norm > .01  # Exercise clipping, not just the unclipped branch.
        for gradient, q in zip(captured[-1], reference.parameters()):
            torch.testing.assert_close(gradient, q.grad, atol=2e-7, rtol=2e-4)
        expected.step()
        for p, q in zip(tiny_train_model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual.state[p][key], expected.state[q][key], atol=1e-7, rtol=1e-5)
            assert actual.state[p]["step"] == expected.state[q]["step"].item()


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_zero_batch_skips_momentum_update_unlike_unconditional_adam(tiny_train_model, tokenizer, variant):
    reference = deepcopy(tiny_train_model)
    actual, expected = warm_moments(tiny_train_model, reference)
    before = [p.detach().clone() for p in tiny_train_model.parameters()]
    prompts, responses = ["Hello"] * 4, ["a test"] * 4
    result = train_batch(tiny_train_model, tokenizer, actual, torch.amp.GradScaler("cuda", enabled=False),
                        prompts, responses, ["42"] * 4, score, SimpleNamespace(group_size=2), VARIANTS[variant])
    assert result["optimizer_step"] == 0
    full_loss(reference, tokenizer, prompts, responses, variant).backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in reference.parameters())
    expected.step()
    for initial, p in zip(before, tiny_train_model.parameters()):
        torch.testing.assert_close(initial, p, atol=0, rtol=0)
        assert actual.state[p]["step"] == 3
    assert any(not torch.equal(p, q) for p, q in zip(tiny_train_model.parameters(), reference.parameters()))
