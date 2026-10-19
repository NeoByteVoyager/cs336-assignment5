"""Full-parameter on-policy GRPO on two GPUs, including 2 x 16 GB GPUs.

Run from the repository root (no torchrun or separate flash-attn install):
    # vLLM must be installed together with its matching PyTorch dependencies.
    python -m pip install 'vllm==0.19.1' accelerate matplotlib pylatexenc 'math-verify[antlr4-13-2]==0.7.0' 'latex2sympy2_extended==1.10.1'
    python scripts/train_grpo_kaggle.py --variant grpo --seeds 0
    python scripts/train_grpo_kaggle.py --variant all --seeds 0 1 2 3
    python scripts/train_grpo_kaggle.py --variant maxrl --seeds 0 --steps 2 --val-examples 16
    python scripts/train_grpo_kaggle.py --plot-only

GPU 0 holds FP32 training parameters/gradients with FP16 autocast. AdamW's
FP32 moments live on CPU; GPU 1 runs a separate FP16 vLLM server by default.
Weights are synchronized through NCCL after every optimizer step, without disk
checkpoints. --inference-backend transformers retains the older slow fallback.
Gradient checkpointing and physical microbatches of one bound GPU memory.
The assignment's 32 logical microbatches of eight are subdivided without
changing the batch of 256, normalization, or one optimizer update per rollout.

Defaults: OLMo-2-0425-1B, r1_zero, 6400/1024 train/val examples, 200 steps,
lr=1e-5, group=8, temperature=top_p=1, max_new_tokens=512, max_grad_norm=1,
AdamW(betas=(0.9, 0.95), eps=1e-8, weight_decay=0).

JSONL metrics/rollouts and aggregate plots are saved locally. W&B is optional.
The same 100 test questions are sampled with data_seed before and after RL;
both sets of full answers and scores are printed and saved as JSONL logs.
Only one FP16 model (~2.6 GB for 1.3B parameters) is kept in best_model/.
Across all variants, seeds, and post-update validation steps, a higher validation
reward replaces its weights in place; ties keep the earlier winner.
No optimizer, scaler, RNG state, or extra model copies are saved.
Each run has its own console/metrics/rollout logs. Use --no-save-model for logs only.
The final comparison ranks variants by mean FINAL validation reward across seeds;
this is distinct from the best single checkpoint selected during training.
A saved model can be loaded for inference, but cannot exactly resume training.
The default session budget is 11.5 hours, reserving time for final inspection.
Unfinished runs are explicitly marked incomplete, not counted as full runs.
Twenty full runs are not guaranteed to fit one session: use separate jobs and
combine their small log directories for --plot-only. No training-state resume.
vLLM token_entropy is sampled-token surprisal (an MC entropy estimate at T=1),
not the full-vocabulary entropy computed by the Transformers fallback.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import csv
import gc
import json
import math
import os
from itertools import groupby
from pathlib import Path
import random
from statistics import mean, stdev
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
    LogitsProcessor, LogitsProcessorList, StopStringCriteria,
)

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.onpolicy_grpo.grpo_components import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_reward,
    compute_rollout_rewards,
)
from cs336_alignment.onpolicy_grpo.utils import tokenize_prompt_and_output
from cs336_alignment.vllm_utils import VLLMServer, start_server, wait_for_server


VARIANTS = {
    "grpo": dict(baseline="mean", advantage_normalizer="std", loss_normalization="sequence"),
    "grpo_constant": dict(baseline="mean", advantage_normalizer="std", loss_normalization="constant"),
    "dr_grpo": dict(baseline="mean", advantage_normalizer="none", loss_normalization="constant"),
    "rft": dict(baseline="none", advantage_normalizer="none", loss_normalization="constant"),
    "maxrl": dict(baseline="mean", advantage_normalizer="mean", loss_normalization="constant"),
}


class CPUAdamW(torch.optim.Optimizer):
    """AdamW with GPU parameters and CPU moments; no persistent CPU master copy.

    Only one parameter and gradient are copied to CPU at a time. FP32 GPU
    parameters work with ordinary GradScaler unscaling and gradient clipping.
    This deliberately trades PCIe/CPU time for GPU and host memory headroom.
    """

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("CPUAdamW does not support closures")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.dtype != torch.float32 or parameter.grad.is_sparse:
                    raise ValueError("CPUAdamW requires dense FP32 parameters and gradients")
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter, device="cpu"),
                                 exp_avg_sq=torch.zeros_like(parameter, device="cpu"))
                state["step"] += 1
                # copy=True also makes the CPU-only verification path safe.
                values = parameter.detach().to("cpu", copy=True)
                gradient = parameter.grad.detach().to("cpu", copy=True)
                mean, variance = state["exp_avg"], state["exp_avg_sq"]
                mean.lerp_(gradient, 1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = variance.sqrt().div_(math.sqrt(1 - beta2 ** state["step"]))
                denominator.add_(group["eps"])
                values.mul_(1 - group["lr"] * group["weight_decay"])
                values.addcdiv_(mean, denominator, value=-group["lr"] / (1 - beta1 ** state["step"]))
                parameter.copy_(values)


def write_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class Tee:
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def console_log(path):
    with Path(path).open("a", encoding="utf-8", buffering=1) as handle:
        with redirect_stdout(Tee(sys.stdout, handle)), redirect_stderr(Tee(sys.stderr, handle)):
            yield


def save_best_model(output_dir, model, tokenizer, candidate):
    """Keep just the highest-reward evaluated RL policy across the whole suite."""
    output_dir = Path(output_dir)
    metadata_path = output_dir / "best_model.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if candidate["val_reward"] <= previous["val_reward"]:
            return False
    # The architecture/shard names stay the same across runs: replace weights
    # in place, without retaining the previous model or a second model directory.
    model.save_pretrained(output_dir / "best_model", max_shard_size="1GB")
    tokenizer.save_pretrained(output_dir / "best_model")
    metadata_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    print(f"New best model: {candidate['variant']}, seed={candidate['seed']}, "
          f"step={candidate['step']}, val_reward={candidate['val_reward']:.4f}", flush=True)
    return True


def read_data(path, count, seed):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if count > len(rows):
        raise ValueError(f"{path}: requested {count} examples, only {len(rows)} available")
    return random.Random(seed).sample(rows, count)


def require_vllm():
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError("Install vllm==0.19.1 with its matching PyTorch dependencies before running") from error
    if installed != "0.19.1":
        raise RuntimeError(f"This weight-sync adapter targets vllm==0.19.1, found {installed}")


def cuda_link_environment(run_dir):
    """Expose the loaded driver to JIT linkers in runtime-only containers."""
    # The policy has initialized CUDA before this call. Use its actual loaded
    # driver, never a toolkit stub or a guessed driver version.
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return {}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        driver = Path(fields[5])
        if not driver.name.startswith("libcuda.so") or not driver.is_file():
            continue
        link_dir = (Path(run_dir) / "cuda_link").resolve()
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / "libcuda.so"
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(driver.resolve())
        existing = os.environ.get("LIBRARY_PATH", "")
        print(f"CUDA JIT linker: {link} -> {driver}", flush=True)
        return {"LIBRARY_PATH": str(link_dir) + (os.pathsep + existing if existing else "")}
    return {}


class VLLMRollout:
    """One server on the rollout GPU, with streamed FP16 NCCL weight updates."""

    def __init__(self, policy, args, seed, run_dir):
        require_vllm()
        self.policy = policy
        self.request_seed = seed
        self.server = VLLMServer(model_id=args.model, gpu=args.rollout_gpu, seed=seed,
                                 port=args.vllm_port, gpu_memory_utilization=args.vllm_memory_fraction)
        log_path = Path(run_dir) / "vllm_server.log"
        try:
            self.server.process = start_server(
                args.model, self.server.host, self.server.port, args.rollout_gpu, seed, "auto", "INFO",
                args.vllm_memory_fraction, dtype="float16", log_path=str(log_path),
                env_overrides=cuda_link_environment(run_dir),
                extra_args=("--max-model-len", str(args.vllm_max_model_len),
                            "--max-num-seqs", str(args.vllm_max_num_seqs),
                            "--max-num-batched-tokens", str(args.vllm_max_batched_tokens),
                            "--generation-config", "vllm", "--enforce-eager"),
            )
            wait_for_server(self.server.base_url, self.server.process, 900)
            self.server.init_weight_sync(str(next(policy.parameters()).device))
        except BaseException:
            self.close()
            print(f"vLLM startup failed; inspect {log_path}", file=sys.stderr)
            # Kaggle background runs expose the notebook console first. Include
            # the subprocess traceback there instead of only its exit code.
            try:
                with log_path.open(encoding="utf-8", errors="replace") as handle:
                    tail = "".join(deque(handle, maxlen=150))
                print(f"--- vLLM server log (last 150 lines) ---\n{tail}",
                      file=sys.stderr, flush=True)
            except OSError as log_error:
                print(f"Could not read vLLM server log: {log_error}", file=sys.stderr)
            raise

    def close(self):
        self.server.stop()
        self.server.weight_sync_group = None

    def sync(self, policy):
        from cs336_alignment.vllm_utils import sync_policy_weights
        self.policy = policy
        # Avoid full-model staging buffers while the inference GPU holds KV cache.
        sync_policy_weights(policy, self.server.base_url, self.server.weight_sync_group,
                            packed=False, transfer_dtype=torch.float16)

    def generate(self, tokenizer, prompts, args, collect_entropy):
        groups = [(prompt, len(list(items))) for prompt, items in groupby(prompts)]
        counts = {count for _, count in groups}
        n = counts.pop() if len(counts) == 1 else 1
        request_prompts = [prompt for prompt, _ in groups] if n > 1 else prompts
        params = dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_new_tokens,
                      n=n, seed=self.request_seed, independent_prompt_seeds=True)
        self.request_seed += len(prompts)
        if args.prompt != "question_only":
            params.update(stop=["</answer>"], include_stop_str_in_output=True)
        if collect_entropy:
            params["logprobs"] = 0
        # n=group_size samples independently within a request. Sending repeated
        # identical prompts with the same explicit seed would collapse the group.
        completions = self.server.generate_completions(
            request_prompts, params, batch_size=args.vllm_request_batch_size)
        if len(completions) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(completions)} responses for {len(prompts)} prompts")
        responses = [item.text for item in completions]
        lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]
        log_probs = [value for item in completions for value in (item.token_log_probs or []) if value is not None]
        if collect_entropy and not log_probs:
            raise RuntimeError("vLLM did not return the requested sampled token log-probabilities")
        entropy = -sum(log_probs) / len(log_probs) if log_probs else None
        return responses, lengths, entropy

    def save_pretrained(self, directory, **kwargs):
        # Export from the HF policy, not a second resident GPU inference model.
        state = {name: tensor.detach().to(device="cpu", dtype=torch.float16) if tensor.is_floating_point()
                 else tensor.detach().cpu() for name, tensor in self.policy.state_dict().items()}
        self.policy.save_pretrained(directory, state_dict=state, **kwargs)
        path = Path(directory) / "config.json"
        config = json.loads(path.read_text())
        config.update(dtype="float16", torch_dtype="float16")
        path.write_text(json.dumps(config, indent=2))


@torch.no_grad()
def sync_weights(policy, rollout_model):
    if isinstance(rollout_model, VLLMRollout):
        rollout_model.sync(policy)
        return
    source = dict(policy.named_parameters())
    for name, parameter in rollout_model.named_parameters():
        parameter.copy_(source[name])
    source_buffers = dict(policy.named_buffers())
    for name, buffer in rollout_model.named_buffers():
        if name in source_buffers:
            buffer.copy_(source_buffers[name])


class RolloutEntropy(LogitsProcessor):
    """Stream entropy during sampling, without retaining vocabulary logits.

    Includes every rollout, even those later pruned for zero advantage, and
    excludes padding steps after a response has finished. Sampling settings
    are temperature=top_p=1, top_k=0, so these scores are the policy logits.
    """

    def __init__(self, tokenizer, prompt_width, batch_size, device, stop_strings):
        self.prompt_width = prompt_width
        self.eos = tokenizer.eos_token_id
        self.stop = StopStringCriteria(tokenizer, stop_strings) if stop_strings else None
        self.active = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.total = torch.zeros((), device=device)
        self.count = torch.zeros((), dtype=torch.long, device=device)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > self.prompt_width:
            if self.eos is not None:
                self.active &= input_ids[:, -1] != self.eos
            if self.stop is not None:
                self.active &= ~self.stop(input_ids, scores)
        log_p = scores.float().log_softmax(-1)
        entropy = -(log_p.exp() * log_p).sum(-1)
        self.total += entropy.masked_fill(~self.active, 0).sum()
        self.count += self.active.sum()
        return scores


@torch.inference_mode()
def generate(model, tokenizer, prompts, args, collect_entropy=False):
    if isinstance(model, VLLMRollout):
        return model.generate(tokenizer, prompts, args, collect_entropy)
    responses, lengths = [], []
    entropy_sum, token_count = 0.0, 0
    device = next(model.parameters()).device
    # A fresh config avoids inheriting top-k, repetition penalties, or sampling
    # defaults from the pretrained checkpoint.
    config = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
        stop_strings=["</answer>"] if args.prompt != "question_only" else None,
    )
    for start in range(0, len(prompts), args.generate_batch_size):
        batch = prompts[start:start + args.generate_batch_size]
        inputs = tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
        if inputs.input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt + requested response exceeds model context; no truncation was applied")
        recorder = (RolloutEntropy(tokenizer, inputs.input_ids.shape[1], len(batch), device, config.stop_strings)
                    if collect_entropy else None)
        output = model.generate(
            **inputs, generation_config=config, tokenizer=tokenizer,
            logits_processor=LogitsProcessorList([recorder]) if collect_entropy else None,
        )
        if collect_entropy:
            entropy_sum += recorder.total.item()
            token_count += recorder.count.item()
        for tokens in output[:, inputs.input_ids.shape[1]:].tolist():
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if args.prompt != "question_only" and "</answer>" in text:
                text = text.split("</answer>", 1)[0] + "</answer>"
            responses.append(text)
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
    return responses, lengths, entropy_sum / max(token_count, 1) if collect_entropy else None


def token_log_probs(model, inputs, labels):
    logits = model(input_ids=inputs, use_cache=False).logits
    return -F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none")


def train_batch(policy, tokenizer, optimizer, scaler, prompts, responses, truths, reward_fn, args, variant):
    raw_rewards, metadata = compute_rollout_rewards(reward_fn, responses, truths)
    advantages, _ = compute_group_normalized_reward(
        raw_rewards, args.group_size, baseline=variant["baseline"],
        advantage_normalizer=variant["advantage_normalizer"],
    )
    active_indices = advantages.nonzero().flatten().tolist()
    metadata.update(active_sequences=len(active_indices), optimizer_step=0,
                    loss=0.0, grad_norm=0.0)
    optimizer.zero_grad(set_to_none=True)
    if not active_indices:
        # In particular do not call scaler.step without any backward/unscale.
        return metadata

    policy.train()
    device = next(policy.parameters()).device
    logical_size = len(prompts) // args.gradient_accumulation_steps
    for logical_start in range(0, len(prompts), logical_size):
        indices = [i for i in active_indices if logical_start <= i < logical_start + logical_size]
        for start in range(0, len(indices), args.microbatch_size):
            selected = indices[start:start + args.microbatch_size]
            # Pad each physical microbatch separately, never to rollout-wide length.
            batch = tokenize_prompt_and_output(
                [prompts[i] for i in selected], [responses[i] for i in selected], tokenizer,
            )
            inputs, labels, mask = (batch[key].to(device) for key in ("input_ids", "labels", "response_mask"))
            if not mask.any(dim=-1).all():
                raise ValueError("A nonzero-advantage response has no response tokens")
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda", cache_enabled=False):
                log_probs = token_log_probs(policy, inputs, labels)
                per_token_loss = -advantages[selected].to(device)[:, None] * log_probs
                loss = aggregate_loss_across_microbatch(
                    per_token_loss, mask, variant["loss_normalization"],
                    normalization_constant=len(prompts) * args.max_new_tokens,
                )
                if variant["loss_normalization"] == "sequence":
                    loss = loss * len(selected) / len(prompts)
            metadata["loss"] += loss.detach().item()
            scaler.scale(loss).backward()
            del loss, per_token_loss, log_probs, inputs, labels, mask

    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
    if not math.isfinite(metadata["loss"]):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite loss; stopping rather than recording a misleading run")
    # GradScaler skips the update when gradients overflow; record it explicitly.
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= old_scale
    if not torch.isfinite(norm) and not scaler.is_enabled():
        raise FloatingPointError("Non-finite gradients")
    metadata.update(grad_norm=norm.item() if torch.isfinite(norm) else None,
                    optimizer_step=int(updated), loss_scale=scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    return metadata


def evaluate(rollout_model, tokenizer, rows, template, reward_fn, args):
    prompts = [template.format(question=row["question"]) for row in rows]
    truths = [row["answer"].split("####")[-1].strip() for row in rows]
    responses, lengths, _ = generate(rollout_model, tokenizer, prompts, args)
    _, rewards = compute_rollout_rewards(reward_fn, responses, truths)
    metrics = {"val_reward": rewards["mean_total_reward"],
               "val_format_reward": rewards["mean_format_reward"],
               "val_response_length": sum(lengths) / len(lengths)}
    samples = [dict(prompt=p, response=r, ground_truth=t, **reward_fn(r, t))
               for p, r, t in zip(prompts, responses, truths)]
    return metrics, samples


def inspect_test_samples(model, tokenizer, rows, template, reward_fn, args, run_dir, phase):
    """Inspect fixed test questions without consuming the training RNG stream."""
    print(f"\n[{phase}] Generating answers for {len(rows)} fixed test questions", flush=True)
    prompts = [template.format(question=row["question"]) for row in rows]
    is_vllm = isinstance(model, VLLMRollout)
    device = torch.device("cpu") if is_vllm else next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.random.default_generator.manual_seed(args.data_seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(args.data_seed)
        if is_vllm:
            previous_seed = model.request_seed
            model.request_seed = args.data_seed
            try:
                responses, lengths, _ = generate(model, tokenizer, prompts, args)
            finally:
                model.request_seed = previous_seed
        else:
            responses, lengths, _ = generate(model, tokenizer, prompts, args)
    total_reward, total_format = 0.0, 0.0
    for index, (row, prompt, response, length) in enumerate(zip(rows, prompts, responses, lengths), 1):
        truth = row["answer"].split("####")[-1].strip()
        scores = reward_fn(response, truth)
        record = dict(phase=phase, sample_id=index, question=row["question"], prompt=prompt,
                      response=response, ground_truth=truth, reference_answer=row["answer"],
                      response_length=length, **scores)
        write_jsonl(Path(run_dir) / f"{phase}_test.jsonl", record)
        print(f"\n{'=' * 80}\n[{phase}] Test {index}/{len(rows)}\n"
              f"Question: {row['question']}\nGround truth: {truth}\n"
              f"Response:\n{response}\n"
              f"Reward: {scores['reward']} | Format reward: {scores['format_reward']}", flush=True)
        total_reward += scores["reward"]
        total_format += scores["format_reward"]
    metrics = dict(examples=len(rows), mean_reward=total_reward / len(rows),
                   mean_format_reward=total_format / len(rows), mean_response_length=sum(lengths) / len(rows))
    print(f"[{phase}] Summary: {json.dumps(metrics)}", flush=True)
    return metrics


def run(args, name, seed, resources, deadline):
    run_started = time.monotonic()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(args.train_gpu)
    sampler = random.Random(seed)
    variant = VARIANTS[name]
    config = {**vars(args), "variant": name, "seed": seed, **variant}
    run_dir = Path(args.output_dir) / f"{args.prompt}_{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.jsonl").exists():
        raise ValueError(f"{run_dir} already has metrics; use a new --output-dir")
    train_rows = read_data(args.train_data, args.train_examples, args.data_seed)
    val_rows = read_data(args.val_data, args.val_examples, args.data_seed)
    # Sample from the full test file, independently of --val-examples. Reusing
    # these rows and data_seed makes before/after and cross-run comparisons easy.
    inspection_rows = read_data(args.val_data, args.inspect_examples, args.data_seed)
    template = (ROOT / "cs336_alignment/prompts" / f"{args.prompt}.prompt").read_text()
    reward_fn = question_only_reward_fn if args.prompt == "question_only" else r1_zero_reward_fn
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa",
        device_map={"": f"cuda:{args.train_gpu}"},
    )
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = CPUAdamW(policy.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.inference_backend == "vllm":
        rollout_model = VLLMRollout(policy, args, seed, run_dir)
        resources.callback(rollout_model.close)
    else:
        rollout_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, attn_implementation="sdpa",
            device_map={"": f"cuda:{args.rollout_gpu}"},
        ).eval()
        rollout_model.requires_grad_(False)
    sync_weights(policy, rollout_model)
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=run_dir.name, config=config)

    def log(record):
        write_jsonl(run_dir / "metrics.jsonl", record)
        print(json.dumps({"run": run_dir.name, **record}), flush=True)
        if wandb_run:
            wandb_run.log(record, step=record["step"])

    phase_started = time.monotonic()
    before = inspect_test_samples(rollout_model, tokenizer, inspection_rows, template,
                                  reward_fn, args, run_dir, "before_rl")
    inspection_seconds = time.monotonic() - phase_started
    (run_dir / "test_inspection_summary.json").write_text(json.dumps({"before_rl": before}, indent=2))
    print(f"Initial validation: {len(val_rows)} examples", flush=True)
    phase_started = time.monotonic()
    metrics, samples = evaluate(rollout_model, tokenizer, val_rows, template, reward_fn, args)
    log({"step": 0, **metrics})
    write_jsonl(run_dir / "val_rollouts.jsonl", dict(step=0, samples=samples))
    validation_seconds = time.monotonic() - phase_started
    setup_seconds = time.monotonic() - run_started
    durations = []
    for step in range(1, args.steps + 1):
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(args.train_gpu)
        if args.inference_backend == "transformers":
            torch.cuda.reset_peak_memory_stats(args.rollout_gpu)
        rows = sampler.sample(train_rows, args.rollout_batch_size // args.group_size)
        prompts = [template.format(question=row["question"]) for row in rows for _ in range(args.group_size)]
        truths = [row["answer"].split("####")[-1].strip() for row in rows for _ in range(args.group_size)]
        # GPU 1 was synchronized after the previous optimizer step (or on load).
        print(f"Step {step}/{args.steps}: generating {len(prompts)} rollouts", flush=True)
        responses, lengths, entropy = generate(rollout_model, tokenizer, prompts, args, collect_entropy=True)
        rollout_seconds = time.monotonic() - started
        print(f"Step {step}/{args.steps}: training", flush=True)
        phase_started = time.monotonic()
        metadata = train_batch(policy, tokenizer, optimizer, scaler, prompts, responses, truths, reward_fn, args, variant)
        train_seconds = time.monotonic() - phase_started
        sync_weights(policy, rollout_model)
        reserve = max(300.0, 1.5 * (inspection_seconds + validation_seconds))
        next_step_seconds = max(durations[-10:], default=time.monotonic() - started)
        stop_for_budget = time.monotonic() + reserve + next_step_seconds >= deadline
        record = dict(step=step, **metadata, token_entropy=entropy, train_response_length=sum(lengths) / len(lengths),
                      token_entropy_estimator="sampled_surprisal" if args.inference_backend == "vllm" else "full_vocabulary",
                      rollout_seconds=rollout_seconds, train_seconds=train_seconds,
                      generated_tokens_per_second=sum(lengths) / max(rollout_seconds, 1e-6),
                      train_peak_gb=torch.cuda.max_memory_allocated(args.train_gpu) / 2**30,
                      rollout_peak_gb=(torch.cuda.max_memory_allocated(args.rollout_gpu) / 2**30
                                       if args.inference_backend == "transformers" else None))
        if step % args.val_every == 0 or step == args.steps or stop_for_budget:
            print(f"Step {step}/{args.steps}: validating", flush=True)
            metrics, samples = evaluate(rollout_model, tokenizer, val_rows, template, reward_fn, args)
            record.update(metrics)
            write_jsonl(run_dir / "val_rollouts.jsonl", dict(step=step, samples=samples))
            if args.save_model:
                record["is_global_best"] = save_best_model(
                    args.output_dir, rollout_model, tokenizer,
                    dict(variant=name, seed=seed, step=step, **metrics,
                         selection_metric="val_reward", run_dir=run_dir.name, config=config),
                )
        if step == 1 or step % args.log_rollouts_every == 0:
            write_jsonl(run_dir / "train_rollouts.jsonl", dict(step=step, samples=[
                dict(prompt=p, response=r, ground_truth=t, **reward_fn(r, t))
                for p, r, t in zip(prompts, responses, truths)]))
        record["step_seconds"] = time.monotonic() - started
        durations.append(record["step_seconds"])
        record["estimated_remaining_seconds"] = mean(durations[-10:]) * (args.steps - step) + inspection_seconds
        record["estimated_200_step_run_hours"] = (setup_seconds + mean(durations[-10:]) * 200 + inspection_seconds) / 3600
        record["time_budget_stop"] = stop_for_budget and step < args.steps
        log(record)
        if stop_for_budget and step < args.steps:
            print(f"Session budget reached at step {step}/{args.steps}; this run is INCOMPLETE. "
                  "Finishing test inspection and keeping the best weights/logs.", flush=True)
            break
    after = inspect_test_samples(rollout_model, tokenizer, inspection_rows, template,
                                 reward_fn, args, run_dir, "after_rl")
    comparison = dict(before_rl=before, after_rl=after, reward_change=after["mean_reward"] - before["mean_reward"])
    (run_dir / "test_inspection_summary.json").write_text(json.dumps(comparison, indent=2))
    print(f"Test inspection before/after: {json.dumps(comparison)}", flush=True)
    if wandb_run:
        wandb_run.finish()
    result = dict(completed=step == args.steps, steps_completed=step, requested_steps=args.steps,
                  wall_seconds=time.monotonic() - run_started)
    (run_dir / "run_result.json").write_text(json.dumps(result, indent=2))
    return result


def write_comparison(output_dir, runs):
    summaries = []
    for config, rows in runs:
        evaluated = [row for row in rows if "val_reward" in row]
        if not evaluated:
            continue
        first, last = evaluated[0], evaluated[-1]
        best = max((row for row in evaluated if row["step"] > 0),
                   key=lambda row: row["val_reward"], default=first)
        summary = {key: config[key] for key in ("variant", "prompt", "seed")}
        summary.update(step=last["step"], completed=last["step"] == config["steps"],
                       initial_val_reward=first["val_reward"], val_reward=last["val_reward"],
                       reward_change=last["val_reward"] - first["val_reward"],
                       val_format_reward=last["val_format_reward"],
                       best_val_reward=best["val_reward"], best_step=best["step"],
                       inspect_before=None, inspect_after=None, inspect_change=None)
        inspection_path = Path(config["_run_dir"]) / "test_inspection_summary.json"
        if inspection_path.exists():
            inspection = json.loads(inspection_path.read_text())
            summary["inspect_before"] = inspection["before_rl"]["mean_reward"]
            if "after_rl" in inspection:
                summary["inspect_after"] = inspection["after_rl"]["mean_reward"]
                summary["inspect_change"] = inspection["reward_change"]
        summaries.append(summary)
    if not summaries:
        return
    output_dir = Path(output_dir)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    aggregates = []
    for prompt, variant in sorted({(row["prompt"], row["variant"]) for row in summaries}):
        group = [row for row in summaries if row["prompt"] == prompt and row["variant"] == variant and row["completed"]]
        if not group:
            continue
        rewards = [row["val_reward"] for row in group]
        aggregate = dict(prompt=prompt, variant=variant, n_seeds=len(group),
                         mean_final_reward=mean(rewards), std_final_reward=stdev(rewards) if len(group) > 1 else 0.0,
                         min_final_reward=min(rewards), max_final_reward=max(rewards),
                         mean_reward_change=mean(row["reward_change"] for row in group))
        for key in ("inspect_before", "inspect_after", "inspect_change"):
            values = [row[key] for row in group if row[key] is not None]
            aggregate[f"mean_{key}"] = mean(values) if values else None
        aggregates.append(aggregate)
    aggregates.sort(key=lambda row: row["mean_final_reward"], reverse=True)
    if aggregates:
        with (output_dir / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    report = ["Comparison of completed runs (mean +/- sample std across available seeds)",
              "Variant | Seeds | Final validation reward | Change from base | 100-test before -> after"]
    for row in aggregates:
        before, after = row["mean_inspect_before"], row["mean_inspect_after"]
        inspection = f"{before:.4f} -> {after:.4f}" if before is not None and after is not None else "pending"
        report.append(f"{row['prompt']}/{row['variant']} | {row['n_seeds']} | "
                      f"{row['mean_final_reward']:.4f} +/- {row['std_final_reward']:.4f} | "
                      f"{row['mean_reward_change']:+.4f} | {inspection}")
    best_path = output_dir / "best_model.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
        report.append(f"Saved best single model: {best['variant']}, seed={best['seed']}, "
                      f"step={best['step']}, val_reward={best['val_reward']:.4f}; weights: best_model/")
    text = "\n".join(report) + "\n"
    (output_dir / "comparison.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def plot_results(output_dir):
    # Keep the latest record for each step if logs were combined externally.
    runs = []
    for path in sorted(Path(output_dir).glob("*/metrics.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            while rows and rows[-1]["step"] >= row["step"]:
                rows.pop()
            rows.append(row)
        config = json.loads((path.parent / "config.json").read_text())
        config["_run_dir"] = str(path.parent)
        runs.append((config, rows))
    if not runs:
        return
    write_comparison(output_dir, runs)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib for plots; summary.csv and JSONL metrics are available.")
        return
    for metric in ("loss", "grad_norm", "token_entropy", "mean_total_reward", "mean_format_reward",
                   "val_reward", "val_format_reward", "val_response_length"):
        fig, axis = plt.subplots(figsize=(8, 5))
        groups = sorted({(config["prompt"], config["variant"]) for config, _ in runs})
        for prompt, variant in groups:
            by_step = {}
            for config, rows in runs:
                if (config["prompt"], config["variant"]) != (prompt, variant):
                    continue
                points = [(row["step"], row[metric]) for row in rows if row.get(metric) is not None]
                for step, value in points:
                    by_step.setdefault(step, []).append(value)
            if not by_step:
                continue
            steps = sorted(by_step)
            line, = axis.plot(steps, [sum(by_step[s]) / len(by_step[s]) for s in steps], label=f"{prompt}/{variant}")
            axis.fill_between(steps, [min(by_step[s]) for s in steps], [max(by_step[s]) for s in steps],
                              color=line.get_color(), alpha=0.15)
        axis.set(xlabel="Rollout / optimizer step", ylabel=metric, title="Mean and min–max across available seeds")
        if axis.lines:
            axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f"{metric}.png", dpi=150)
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="grpo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--inference-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-memory-fraction", type=float, default=0.75)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-max-batched-tokens", type=int, default=4096)
    parser.add_argument("--vllm-request-batch-size", type=int, default=256)
    parser.add_argument("--max-hours", type=float, default=11.5,
                        help="Whole-process budget including all runs; leaves margin under Kaggle's 12h limit")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", choices=["r1_zero", "r1_zero_three_shot_gsm8k", "question_only"], default="r1_zero")
    parser.add_argument("--train-data", default=str(ROOT / "data/gsm8k/train.jsonl"))
    parser.add_argument("--val-data", default=str(ROOT / "data/gsm8k/test.jsonl"))
    parser.add_argument("--output-dir", default="/kaggle/working/grpo_results")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-examples", type=int, default=6400)
    parser.add_argument("--val-examples", type=int, default=1024)
    parser.add_argument("--inspect-examples", type=int, default=100,
                        help="Fixed test questions printed and logged before and after RL")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                        help="Logical accumulation count; each logical microbatch is subdivided for memory")
    parser.add_argument("--microbatch-size", type=int, default=1, help="Physical training microbatch size")
    parser.add_argument("--generate-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep only the best FP16 model across all variants/seeds/validation steps")
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--rollout-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--wandb-project")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    for key in ("steps", "train_examples", "val_examples", "inspect_examples", "rollout_batch_size", "group_size",
                "gradient_accumulation_steps", "microbatch_size", "generate_batch_size", "max_new_tokens",
                "val_every", "log_rollouts_every", "cpu_threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.group_size < 2:
        parser.error("--group-size must be at least 2 for sample std normalization")
    if args.rollout_batch_size % args.group_size or args.rollout_batch_size % args.gradient_accumulation_steps:
        parser.error("Rollout batch size must be divisible by group size and logical accumulation steps")
    if args.microbatch_size > args.rollout_batch_size // args.gradient_accumulation_steps:
        parser.error("Physical microbatch size must not exceed logical microbatch size")
    if args.train_examples < args.rollout_batch_size // args.group_size:
        parser.error("Training dataset must have enough distinct prompts for one rollout batch")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("Learning rate and grad norm must be positive")
    if not 0 < args.max_hours < 12:
        parser.error("--max-hours must be positive and less than 12")
    if not 0 < args.vllm_memory_fraction < 1:
        parser.error("--vllm-memory-fraction must be between zero and one")
    if min(args.vllm_max_num_seqs, args.vllm_max_batched_tokens, args.vllm_request_batch_size) <= 0:
        parser.error("vLLM batching limits must be positive")
    if args.vllm_max_model_len <= args.max_new_tokens:
        parser.error("vLLM context length must leave room for the prompt")
    return args


def main():
    session_started = time.monotonic()
    args = parse_args()
    if args.plot_only:
        plot_results(args.output_dir)
        return
    if args.train_gpu == args.rollout_gpu or min(args.train_gpu, args.rollout_gpu) < 0:
        raise ValueError("Choose two distinct nonnegative GPU indices")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(args.train_gpu, args.rollout_gpu):
        raise RuntimeError("Two CUDA GPUs are required; select a dual-GPU Kaggle accelerator")
    torch.set_num_threads(args.cpu_threads)
    for gpu in (args.train_gpu, args.rollout_gpu):
        props = torch.cuda.get_device_properties(gpu)
        print(f"GPU {gpu}: {props.name}, {props.total_memory / 2**30:.1f} GiB", flush=True)
    print(f"Backend={args.inference_backend}; FP32 train parameters + FP16 AMP; CPU AdamW; "
          f"session budget={args.max_hours} hours.", flush=True)
    if args.inference_backend == "vllm":
        require_vllm()
        print("token_entropy is sampled-token surprisal, an MC entropy estimate at temperature 1.", flush=True)
    deadline = session_started + args.max_hours * 3600
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    last_run_seconds = None
    try:
        for name in names:
            for seed in args.seeds:
                run_dir = Path(args.output_dir) / f"{args.prompt}_{name}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                if (run_dir / "metrics.jsonl").exists():
                    result_path = run_dir / "run_result.json"
                    if args.skip_completed and result_path.exists() and json.loads(result_path.read_text())["completed"]:
                        saved = json.loads((run_dir / "config.json").read_text())
                        for key in ("model", "prompt", "steps", "learning_rate", "group_size", "rollout_batch_size",
                                    "max_new_tokens", "data_seed", "train_examples", "val_examples", "inference_backend"):
                            if saved.get(key) != getattr(args, key):
                                raise ValueError(f"Completed run configuration differs ({key}); use a new --output-dir")
                        print(f"Skipping completed run {run_dir.name}", flush=True)
                        continue
                    raise ValueError(f"{run_dir} has an incomplete run; optimizer state was not saved. "
                                     "Restart that variant/seed in a new --output-dir.")
                if time.monotonic() + (1.15 * last_run_seconds if last_run_seconds else 600) >= deadline:
                    print("Not enough session budget to start another run; generate comparison and exit.", flush=True)
                    plot_results(args.output_dir)
                    return
                with console_log(run_dir / "console.log"), ExitStack() as resources:
                    print(f"Starting {name}, seed={seed}", flush=True)
                    result = run(args, name, seed, resources, deadline)
                last_run_seconds = result["wall_seconds"]
                gc.collect()
                for gpu in ((args.train_gpu,) if args.inference_backend == "vllm" else (args.train_gpu, args.rollout_gpu)):
                    with torch.cuda.device(gpu):
                        torch.cuda.empty_cache()
                plot_results(args.output_dir)
                if not result["completed"]:
                    return
        plot_results(args.output_dir)
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM: keep --microbatch-size 1. For vLLM lower --vllm-max-num-seqs to 16; "
              "for Transformers lower --generate-batch-size to 1. "
              "Keep rollout batch/group size and generation length fixed for comparable experiments.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
