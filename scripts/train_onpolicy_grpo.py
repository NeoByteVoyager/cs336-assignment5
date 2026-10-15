"""Full-parameter on-policy GRPO for two T4 GPUs (vLLM 0.19.1).

From the repository root, in an environment with the project dependencies and vLLM:
    python scripts/train_onpolicy_grpo.py --output-dir experiments/grpo-seed0

Small integration run:
    python scripts/train_onpolicy_grpo.py --num-rollout-steps 2 \
        --rollout-batch-size 8 --group-size 4 --n-val-examples 16 --eval-every 1 \
        --output-dir /tmp/grpo-smoke

Defaults: GPU 0 trains FP32 parameters using FP16 autocast + GradScaler; GPU 1
runs FP16 vLLM. CPU-offloaded, ordinary AdamW saves training VRAM without
quantizing optimizer states. Allow roughly 20+ GB of host RAM for its parameters,
gradients and moments, in addition to the Python runtime/model-loading overhead.
Use --optimizer adamw8bit (requires bitsandbytes) for an alternative, or --optimizer
adamw on a larger training GPU. Checkpoints contain model/tokenizer/config only;
this script does not implement resuming optimizer state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

# Also support `python scripts/train_onpolicy_grpo.py` without installing the repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.onpolicy_grpo.grpo_components import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_reward,
    compute_policy_gradient_loss,
    compute_rollout_rewards,
)
from cs336_alignment.vllm_utils import VLLMServer, wait_for_server


@dataclass
class Example:
    prompt: str
    prompt_ids: list[int]
    ground_truth: str


@dataclass
class Rollout:
    example: Example
    text: str
    token_ids: list[int]
    finish_reason: str | None


def load_examples(path, template, tokenizer, limit, seed):
    """Select a reproducible subset; tokenize prompts without implicit BOS/EOS."""
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < limit:
        raise ValueError(f"{path}: requested {limit} examples, found {len(rows)}")
    random.Random(seed).shuffle(rows)
    examples = []
    for row in rows[:limit]:
        if "####" not in row["answer"]:
            raise ValueError(f"{path}: expected a GSM8K answer containing ####")
        prompt = template.format(question=row["question"])
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("An empty prompt cannot predict the first response token")
        examples.append(Example(prompt, ids, row["answer"].rsplit("####", 1)[1].strip()))
    return examples


class CPUAdamW(torch.optim.Optimizer):
    """Ordinary FP32 AdamW on CPU shadow parameters, compatible with GradScaler.

    Public param_groups contain the GPU parameters so scaler.unscale_ and clipping
    act on the actual training gradients. The inner optimizer owns CPU parameters
    and moments. Blocking copies make update ordering explicit.
    """

    def __init__(self, parameters, lr, betas=(0.9, 0.95)):
        parameters = list(parameters)
        super().__init__(parameters, dict(lr=lr, betas=betas, weight_decay=0.0))
        self.cpu_parameters = [
            torch.nn.Parameter(p.detach().to(device="cpu", copy=True)) for p in parameters
        ]
        self.cpu_optimizer = torch.optim.AdamW(
            self.cpu_parameters, lr=lr, betas=betas, weight_decay=0.0, foreach=False
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("CPUAdamW does not support closures")
        for group, cpu_group in zip(self.param_groups, self.cpu_optimizer.param_groups):
            cpu_group["lr"] = group["lr"]
            for gpu_p, cpu_p in zip(group["params"], cpu_group["params"]):
                cpu_p.grad = None if gpu_p.grad is None else gpu_p.grad.detach().cpu()
        self.cpu_optimizer.step()
        for gpu_p, cpu_p in zip(self.param_groups[0]["params"], self.cpu_parameters):
            gpu_p.copy_(cpu_p)
        self.cpu_optimizer.zero_grad(set_to_none=True)


def http_json(method, url, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return json.loads(data) if data else {}


class T4VLLMServer(VLLMServer):
    """Script-local FP16 startup/sync; leave the supplied BF16 helpers untouched."""

    def start_t4(self, args, log_path):
        # Never terminate another process just because it uses the requested port.
        with socket.socket() as probe:
            probe.bind((self.host, self.port))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_selector = str(self.gpu)
        if visible is not None:
            devices = visible.split(",")
            if self.gpu >= len(devices):
                raise ValueError("inference GPU is outside CUDA_VISIBLE_DEVICES")
            gpu_selector = devices[self.gpu].strip()
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=gpu_selector,
            VLLM_SERVER_DEV_MODE="1",
            VLLM_LOGGING_LEVEL="INFO",
        )
        command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_id, "--host", self.host, "--port", str(self.port),
            "--dtype", "float16", "--tensor-parallel-size", "1",
            "--seed", str(self.seed), "--enforce-eager",
            "--attention-backend", args.vllm_attention_backend,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--max-num-seqs", str(args.vllm_max_num_seqs),
            "--max-num-batched-tokens", str(args.vllm_max_num_batched_tokens),
            "--enable-prefix-caching", "--generation-config", "vllm",
            "--weight-transfer-config", json.dumps({"backend": "nccl"}),
        ]
        with Path(log_path).open("w", encoding="utf-8") as log:
            self.process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            wait_for_server(self.base_url, self.process, self.startup_timeout)
        except Exception as exc:
            raise RuntimeError(f"vLLM failed to start; inspect {log_path}") from exc

    @torch.no_grad()
    def sync_fp16_weights(self, model):
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine,
        )

        if self.weight_sync_group is None:
            raise RuntimeError("Initialize the NCCL weight sync group first")
        weights = list(model.named_parameters())
        update_info = {
            "names": [name for name, _ in weights],
            "dtype_names": ["float16"] * len(weights),
            "shapes": [list(p.shape) for _, p in weights],
            "packed": False,
        }
        torch.cuda.set_device(next(model.parameters()).device)
        http_json("POST", self.base_url + "/pause")
        # One tensor at a time avoids a second complete GPU copy of the policy.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                http_json, "POST", self.base_url + "/update_weights",
                {"update_info": update_info},
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(weights),
                trainer_args=NCCLTrainerSendWeightsArgs(
                    group=self.weight_sync_group, packed=False,
                    post_iter_func=lambda item: item[1].detach().to(torch.float16).contiguous(),
                ),
            )
            future.result()
        http_json("POST", self.base_url + "/reset_prefix_cache")
        http_json("POST", self.base_url + "/resume")


def generate_rollouts(server, examples, n, args, seed):
    """Use exact sampled token IDs rather than decoding then re-tokenizing actions."""
    rollouts = []
    for start in range(0, len(examples), args.generation_batch_size):
        batch = examples[start:start + args.generation_batch_size]
        payload = {
            "model": args.model,
            "prompt": [example.prompt_ids for example in batch],
            "n": n, "temperature": args.temperature, "top_p": 1.0, "top_k": -1,
            "max_tokens": args.max_tokens, "seed": seed + start,
            "return_token_ids": True,
        }
        if args.reward_type == "r1_zero":
            payload.update(stop=["</answer>"], include_stop_str_in_output=True)
        response = http_json("POST", server.base_url + "/v1/completions", payload, 3600)
        choices = sorted(response["choices"], key=lambda choice: choice["index"])
        expected = len(batch) * n
        if [c["index"] for c in choices] != list(range(expected)):
            raise RuntimeError("Unexpected vLLM choice indices; cannot safely align groups")
        for index, choice in enumerate(choices):
            ids = choice.get("token_ids")
            if ids is None:
                raise RuntimeError("vLLM did not return token_ids; use the project's vLLM 0.19.1")
            if not ids:
                raise RuntimeError("vLLM returned an empty token trajectory")
            if len(batch[index // n].prompt_ids) + len(ids) > args.max_model_len:
                raise RuntimeError("Generated trajectory exceeds max_model_len")
            rollouts.append(Rollout(
                batch[index // n], choice["text"], ids, choice.get("finish_reason"),
            ))
    return rollouts


def collate_rollouts(rollouts, pad_token_id, device):
    lengths = [len(r.example.prompt_ids) + len(r.token_ids) for r in rollouts]
    tokens = torch.full((len(rollouts), max(lengths)), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(tokens, dtype=torch.bool)
    mask = torch.zeros((len(rollouts), max(lengths) - 1), dtype=torch.bool)
    for i, (r, length) in enumerate(zip(rollouts, lengths)):
        tokens[i, :length] = torch.tensor(r.example.prompt_ids + r.token_ids)
        attention[i, :length] = True
        mask[i, len(r.example.prompt_ids) - 1:length - 1] = True
    return {
        "input_ids": tokens[:, :-1].to(device),
        "labels": tokens[:, 1:].to(device),
        "attention_mask": attention[:, :-1].to(device),
        "response_mask": mask.to(device),
    }


def chunk_log_probs_and_entropy(logits, labels):
    log_probs = logits.float().log_softmax(dim=-1)
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return selected, entropy


def score_microbatch(model, batch, chunk_size, amp_enabled):
    # Checkpoint vocabulary-sized log-softmax as well as transformer layers.
    # Otherwise FP32 softmax intermediates can dominate memory on a 16 GB T4.
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
    probs, entropies = [], []
    for start in range(0, logits.shape[1], chunk_size):
        result = checkpoint(
            chunk_log_probs_and_entropy,
            logits[:, start:start + chunk_size], batch["labels"][:, start:start + chunk_size],
            use_reentrant=False,
        )
        probs.append(result[0])
        entropies.append(result[1])
    return torch.cat(probs, dim=1), torch.cat(entropies, dim=1)


def train_step(model, tokenizer, optimizer, scaler, rollouts, reward_fn, args):
    """Exactly one successful optimizer update per rollout batch.

    On FP16 gradient overflow, recompute on the SAME data with a smaller scale;
    no stale-data update or silently skipped training step is reported.
    """
    device = next(model.parameters()).device
    rewards, reward_metrics = compute_rollout_rewards(
        reward_fn, [r.text for r in rollouts], [r.example.ground_truth for r in rollouts]
    )
    advantages, _ = compute_group_normalized_reward(
        rewards, args.group_size, advantage_eps=args.advantage_eps,
    )
    advantages = advantages.to(device)
    model.train()
    for attempt in range(args.max_amp_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_loss, entropy_sum, token_count = 0.0, 0.0, 0
        for start in range(0, len(rollouts), args.microbatch_size):
            micro = rollouts[start:start + args.microbatch_size]
            batch = collate_rollouts(micro, tokenizer.pad_token_id, device)
            log_probs, entropy = score_microbatch(model, batch, args.logprob_chunk_size, device.type == "cuda")
            per_token_loss, _ = compute_policy_gradient_loss(
                advantages[start:start + len(micro)], log_probs,
            )
            loss = aggregate_loss_across_microbatch(per_token_loss, batch["response_mask"])
            # The final microbatch may be smaller. Weight by ORIGINAL batch size.
            loss = loss * (len(micro) / len(rollouts))
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite forward loss; check FP16 model activations")
            scaler.scale(loss).backward()
            batch_loss += loss.detach().item()
            entropy_sum += entropy.masked_fill(~batch["response_mask"], 0).sum().item()
            token_count += batch["response_mask"].sum().item()
            del batch, log_probs, entropy, per_token_loss, loss
        scaler.unscale_(optimizer)
        # GradScaler decides whether non-finite gradients require skipping step().
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf"), error_if_nonfinite=False,
        )
        old_scale = scaler.get_scale()
        if not torch.isfinite(grad_norm).item():
            # A norm can overflow even when every gradient element is finite.
            # Do not rely only on GradScaler's elementwise inf/nan detection.
            scaler.update(new_scale=old_scale * 0.5)
            optimizer.zero_grad(set_to_none=True)
            print(f"Non-finite gradient norm: retry {attempt + 1}, scale={scaler.get_scale()}", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            return {
                "train/loss": batch_loss, "train/grad_norm": grad_norm.item(),
                "train/token_entropy": entropy_sum / max(token_count, 1),
                "train/reward": reward_metrics["mean_total_reward"],
                "train/format_reward": reward_metrics["mean_format_reward"],
                "train/response_length": sum(len(r.token_ids) for r in rollouts) / len(rollouts),
                "train/zero_advantage_fraction": (advantages == 0).float().mean().item(),
                "train/amp_scale": scaler.get_scale(), "train/amp_retries": attempt,
            }
        print(f"FP16 gradient overflow: retry {attempt + 1}, scale={scaler.get_scale()}", flush=True)
    raise FloatingPointError("Exceeded AMP retries without a successful optimizer update")


def evaluate(server, examples, reward_fn, args):
    # Fixed evaluation seed avoids consuming the training sampler's RNG stream.
    rollouts = generate_rollouts(server, examples, 1, args, args.seed + 1_000_000)
    rewards, metadata = compute_rollout_rewards(
        reward_fn, [r.text for r in rollouts], [r.example.ground_truth for r in rollouts]
    )
    return {
        "val/reward": rewards.mean().item(),
        "val/format_reward": metadata["mean_format_reward"],
        "val/response_length": sum(len(r.token_ids) for r in rollouts) / len(rollouts),
        "val/length_finish_fraction": sum(r.finish_reason == "length" for r in rollouts) / len(rollouts),
    }, rollouts


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def save_samples(path, rollouts, step, split, reward_fn, limit):
    for r in rollouts[:limit]:
        append_jsonl(path, {
            "step": step, "split": split, "prompt": r.example.prompt,
            "response": r.text, "ground_truth": r.example.ground_truth,
            "response_length": len(r.token_ids), "finish_reason": r.finish_reason,
            **reward_fn(r.text, r.example.ground_truth),
        })


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/gsm8k/train.jsonl")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/gsm8k/test.jsonl")
    parser.add_argument("--prompt-path", type=Path, default=ROOT / "cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--reward-type", choices=["r1_zero", "question_only"], default="r1_zero")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/grpo-onpolicy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256, help="Number of responses, not questions")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=["cpu-adamw", "adamw", "adamw8bit"], default="cpu-adamw")
    parser.add_argument("--max-amp-retries", type=int, default=8)
    parser.add_argument("--logprob-chunk-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--generation-batch-size", type=int, default=4, help="Prompts per HTTP request")
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--inference-gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-attention-backend", default="TRITON_ATTN")
    parser.add_argument("--vllm-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--log-rollout-count", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=0, help="0 saves only the final model")
    parser.add_argument("--wandb-project", default=None, help="Optional; local JSONL logs are always written")
    parser.add_argument("--run-name", default=None)
    return parser


def validate_args(args):
    for key in (
        "n_train_examples", "n_val_examples", "num_rollout_steps", "rollout_batch_size",
        "microbatch_size", "logprob_chunk_size", "max_tokens", "max_model_len",
        "generation_batch_size", "eval_every", "log_rollouts_every", "log_rollout_count",
        "vllm_max_num_seqs", "vllm_max_num_batched_tokens",
    ):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.group_size < 2 or args.rollout_batch_size % args.group_size:
        raise ValueError("group-size must be >= 2 and divide rollout-batch-size")
    if args.n_train_examples < args.rollout_batch_size // args.group_size:
        raise ValueError("Training subset is smaller than one batch of prompts")
    for key in ("learning_rate", "max_grad_norm", "advantage_eps", "temperature"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < args.vllm_memory_utilization < 1:
        raise ValueError("vllm-memory-utilization must be between 0 and 1")
    if args.max_tokens >= args.max_model_len:
        raise ValueError("max-model-len must leave space for the prompt")
    if args.save_every < 0 or args.max_amp_retries < 0:
        raise ValueError("save-every and max-amp-retries cannot be negative")
    if args.train_gpu < 0 or args.inference_gpu < 0 or args.train_gpu == args.inference_gpu:
        raise ValueError("Use two distinct, nonnegative GPU indices")


def save_model(model, tokenizer, directory, config):
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True, max_shard_size="1GB")
    tokenizer.save_pretrained(directory)
    (directory / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main(argv=None):
    args = make_parser().parse_args(argv)
    validate_args(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(args.train_gpu, args.inference_gpu):
        raise RuntimeError("This script requires two visible CUDA GPUs")
    from importlib.metadata import version
    if version("vllm").split("+")[0] != "0.19.1":
        raise RuntimeError("Use vllm==0.19.1: the weight-transfer API is version-specific")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.train_gpu)
    device = torch.device(f"cuda:{args.train_gpu}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    template = args.prompt_path.read_text(encoding="utf-8")
    train_examples = load_examples(args.train_data, template, tokenizer, args.n_train_examples, args.seed)
    val_examples = load_examples(args.val_data, template, tokenizer, args.n_val_examples, args.seed + 1)
    longest = max(len(e.prompt_ids) for e in train_examples + val_examples)
    if longest + args.max_tokens > args.max_model_len:
        raise ValueError(f"Longest prompt ({longest}) + max-tokens exceeds max-model-len; increase it")
    # Refuse to mix separate runs or overwrite an existing model.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["gradient_accumulation_steps"] = math.ceil(args.rollout_batch_size / args.microbatch_size)
    config["torch_version"] = torch.__version__
    config["vllm_version"] = version("vllm")
    config["transformers_version"] = version("transformers")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)
    server = T4VLLMServer(
        model_id=args.model, gpu=args.inference_gpu, seed=args.seed, port=args.port,
        gpu_memory_utilization=args.vllm_memory_utilization,
    )
    wandb_run = None
    try:
        server.start_t4(args, args.output_dir / "vllm.log")
        # FP32 master parameters are essential: GradScaler cannot unscale FP16 grads.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="sdpa",
        ).to(device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Dropout changes the scoring policy relative to the inference policy.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        if args.optimizer == "cpu-adamw":
            optimizer = CPUAdamW(model.parameters(), args.learning_rate)
        elif args.optimizer == "adamw8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                weight_decay=0.0, foreach=False,
            )
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        server.init_weight_sync(str(device))
        if args.wandb_project:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=config)
        reward_fn = r1_zero_reward_fn if args.reward_type == "r1_zero" else question_only_reward_fn
        rng = random.Random(args.seed + 2)
        started = time.monotonic()

        def log(step, metrics):
            record = {"step": step, "elapsed_seconds": time.monotonic() - started, **metrics}
            append_jsonl(args.output_dir / "metrics.jsonl", record)
            print(json.dumps(record, allow_nan=False), flush=True)
            if wandb_run is not None:
                wandb_run.log(record, step=step)

        def validate(step):
            metrics, samples = evaluate(server, val_examples, reward_fn, args)
            save_samples(args.output_dir / "rollouts.jsonl", samples, step, "val", reward_fn, args.log_rollout_count)
            return metrics

        server.sync_fp16_weights(model)
        log(0, validate(0))
        for step in range(1, args.num_rollout_steps + 1):
            step_started = time.monotonic()
            # vLLM is already synchronized: initial sync or the previous update's sync.
            batch = rng.sample(train_examples, args.rollout_batch_size // args.group_size)
            rollouts = generate_rollouts(server, batch, args.group_size, args, args.seed + step * 100_000)
            metrics = train_step(model, tokenizer, optimizer, scaler, rollouts, reward_fn, args)
            if step == 1 or step % args.log_rollouts_every == 0:
                save_samples(args.output_dir / "rollouts.jsonl", rollouts, step, "train", reward_fn, args.log_rollout_count)
            # Synchronize AFTER updating: both validation and next rollouts see new weights.
            server.sync_fp16_weights(model)
            if step % args.eval_every == 0 or step == args.num_rollout_steps:
                metrics.update(validate(step))
            metrics["train/step_seconds"] = time.monotonic() - step_started
            metrics["train/peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            log(step, metrics)
            if args.save_every and step % args.save_every == 0:
                save_model(model, tokenizer, args.output_dir / f"checkpoint-{step}", config)
        save_model(model, tokenizer, args.output_dir / "final", config)
    finally:
        server.stop()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
