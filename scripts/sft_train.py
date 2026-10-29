import math
import os
import time
import torch
import argparse
import json
import sys
import traceback
from pathlib import Path
from contextlib import nullcontext, contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from huggingface_hub import HfApi, get_token
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from cs336_alignment.sft.data_loader import (
    SFTDataset,
)
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


class Tee:
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.log_file.write(text)
        self.log_file.flush()
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class LocalLogs:
    def __init__(self, directory):
        self.directory = directory
        self.summary = {"status": "running"}

    def save_json(self, name, values):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def log_metrics(self, step, metrics):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "step": step, **metrics}
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def log_eval(self, step, metrics):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "step": step, **metrics}
        with (self.directory / "eval_metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        self.save_json("eval_latest.json", record)


@contextmanager
def local_logging(args, world_size):
    # torchrun supplies RANK before the process group is initialized.
    if int(os.environ.get("RANK", "0")) != 0:
        yield None
        return
    root = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / "logs"
    directory = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    logs = LocalLogs(directory)
    logs.save_json("config.json", {**vars(args), "world_size": world_size,
        "per_device_micro_batch_size": args.batch_size // (world_size * args.gradient_accumulation_steps)})
    logs.summary["started_at"] = datetime.now(timezone.utc).isoformat()
    logs.save_json("summary.json", logs.summary)
    with (directory / "train.log").open("a", encoding="utf-8") as file:
        with redirect_stdout(Tee(sys.stdout, file)), redirect_stderr(Tee(sys.stderr, file)):
            print(f"Local logs: {directory.resolve()}", flush=True)
            try:
                yield logs
            except BaseException as error:
                logs.summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                traceback.print_exc(file=file)
                raise
            else:
                logs.summary["status"] = "completed"
            finally:
                logs.summary["finished_at"] = datetime.now(timezone.utc).isoformat()
                logs.save_json("summary.json", logs.summary)


def prepare_hf_repo(repo_id, private=True):
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError:
            token = get_token()
        else:
            try:
                token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                raise RuntimeError(
                    "Add/enable the HF_TOKEN secret in Kaggle Add-ons > Secrets. "
                    "Use a Hugging Face token with write access to the target repository."
                ) from None
    if not token:
        raise RuntimeError("HF_TOKEN is missing; training has not started.")
    api = HfApi(token=token)
    api.whoami()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api


def login_wandb():
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError:
            pass  # Outside Kaggle, use existing W&B credentials.
        else:
            try:
                api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
            except Exception:
                raise RuntimeError(
                    "Enable Internet and add/enable the WANDB_API_KEY secret "
                    "in Kaggle Add-ons > Secrets before running training."
                ) from None
            if not api_key:
                raise RuntimeError("The Kaggle WANDB_API_KEY secret is empty.")
    if not wandb.login(key=api_key, verify=True, timeout=30):
        raise RuntimeError("W&B login failed; training has not started.")


def get_model_and_tokenizer(
    model_id: str,
    task_type: str="CAUSAL_LM",
    r: int=8,
    lora_alpha: int=32,
    lora_dropout: float=0.05,
    target_modules: str="all-linear",
    bias: str="none",
    device_index: int=0,
):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        dtype=torch.float16,
        device_map={"": device_index},
        attn_implementation="sdpa",
    )

    model.config.use_cache = False

    # 1. freeze the base model
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 2. add LoRA config
    config = LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )

    # 3. get lora model
    model = get_peft_model(model, config)
    if not dist.is_initialized() or dist.get_rank() == 0:
        model.print_trainable_parameters()

    return tokenizer, model

def save_checkpoint(model, tokenizer, args, step, run_id, hf_api=None):
    relative_path = f"checkpoints/{run_id}/step-{step:06d}"
    directory = Path(args.output_dir) / relative_path
    directory.mkdir(parents=True, exist_ok=False)
    unwrapped_model = model.module if isinstance(model, DDP) else model
    print(f"Saving checkpoint at step {step}: {directory}", flush=True)
    unwrapped_model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    (directory / "checkpoint_info.json").write_text(json.dumps({
        "step": step, "checkpoint_type": "adapter_only",
        "base_model": args.model_id, "config": vars(args),
    }, indent=2) + "\n", encoding="utf-8")
    if hf_api is None:
        return None
    print(f"Uploading checkpoint to {args.hf_repo_id}/{relative_path}", flush=True)
    try:
        commit = hf_api.upload_folder(
            repo_id=args.hf_repo_id, repo_type="model", folder_path=str(directory),
            path_in_repo=relative_path, commit_message=f"Save SFT adapter checkpoint at step {step}",
        )
    except Exception:
        print(f"Checkpoint upload failed; local files retained at {directory}", flush=True)
        raise
    print(f"Checkpoint uploaded: {commit.commit_url}", flush=True)
    return commit.commit_url


@torch.no_grad()
def evaluate(model, dataloader):
    # Unwrap DDP: uneven eval shards must not trigger per-forward collectives.
    eval_model = model.module if isinstance(model, DDP) else model
    device = next(eval_model.parameters()).device
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    try:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = eval_model(input_ids).logits
            nll = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum",
            )
            totals[0] += nll.double()
            totals[1] += (labels != -100).sum()
            totals[2] += input_ids.shape[0]
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = torch.tensor(time.perf_counter() - start, device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        nll_sum, token_count, sequences = totals.tolist()
        if token_count == 0:
            raise ValueError("Evaluation dataset has no valid target tokens.")
        loss = nll_sum / token_count
        return {"eval/loss": loss, "eval/perplexity": math.exp(loss) if loss < 700 else float("inf"),
                "eval/tokens": int(token_count), "eval/sequences": int(sequences),
                "time/eval_seconds": duration.item()}
    finally:
        model.train(was_training)


def sft_train(
    model: AutoModelForCausalLM,
    optimizer: torch.optim,
    dataloader: DataLoader,
    gradient_accumulation_steps: int,
    clipping: float,
    train_steps: int,
    wandb_run=None,
    local_logs=None,
    save_every=0,
    checkpoint_callback=None,
    eval_dataloader=None,
    eval_every=50,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    distributed = dist.is_initialized()
    is_main = not distributed or dist.get_rank() == 0
    world_size = dist.get_world_size() if distributed else 1
    device = next(model.parameters()).device

    def elapsed_seconds(start):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - start, device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item()

    sheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(0.03 * train_steps),
        num_training_steps=train_steps
    )

    step = 0
    total_loss = 0
    batches_processed = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if distributed:
        dist.barrier()
    epoch_start = time.perf_counter()
    step_start = epoch_start
    planned_steps = min(train_steps, math.ceil(len(dataloader) / gradient_accumulation_steps))
    step_durations = []
    best_eval_loss = math.inf
    for index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        group_start = (index // gradient_accumulation_steps) * gradient_accumulation_steps
        accumulation_steps = min(gradient_accumulation_steps, len(dataloader) - group_start)
        update_step = (index + 1) % gradient_accumulation_steps == 0 or index + 1 == len(dataloader)
        sync_context = model.no_sync() if isinstance(model, DDP) and not update_step else nullcontext()
        with sync_context:
            logits = model(input_ids).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1)
            ) / accumulation_steps
            total_loss += loss.item()
            loss.backward()
        batches_processed += 1

        if update_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            sheduler.step()

            optimizer.zero_grad()
            step += 1
            step_seconds = elapsed_seconds(step_start)
            step_durations.append(step_seconds)
            if step == 1:
                first_step_time = step_seconds
                if is_main:
                    print(f"First step time: {first_step_time:.2f}s", flush=True)
                if wandb_run is not None:
                    wandb_run.summary["time/first_step_seconds"] = first_step_time
                if local_logs is not None:
                    local_logs.summary["time/first_step_seconds"] = first_step_time
            loss_metric = torch.tensor(total_loss, device=device)
            if distributed:
                dist.all_reduce(loss_metric, op=dist.ReduceOp.SUM)
            total_loss = loss_metric.item() / world_size
            # Exclude the warm-up step from ETA once later timings are available.
            recent_times = step_durations[1:][-20:] or step_durations
            eta_seconds = sum(recent_times) / len(recent_times) * (planned_steps - step)
            metrics = {
                    "train/loss": total_loss,
                    "train/lr": learning_rate,
                    "train/grad_norm": grad_norm.item(),
                    "train/effective_batch_size": input_ids.shape[0] * accumulation_steps * world_size,
                    "time/step_seconds": step_seconds,
                    "time/eta_seconds": eta_seconds,
                }
            if is_main:
                print(f"step {step}/{planned_steps} | loss {total_loss:.4f} | lr {learning_rate:.3e} | "
                      f"grad_norm {metrics['train/grad_norm']:.3f} | {step_seconds:.2f}s/step | "
                      f"ETA {eta_seconds / 60:.1f} min", flush=True)
            if eval_dataloader is not None and (step % eval_every == 0 or step == planned_steps):
                if is_main:
                    print(f"Evaluating at step {step}...", flush=True)
                eval_metrics = evaluate(model, eval_dataloader)
                metrics.update(eval_metrics)
                eval_summary = {"eval/latest_step": step, **eval_metrics}
                if eval_metrics["eval/loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval/loss"]
                    eval_summary.update({"eval/best_loss": best_eval_loss, "eval/best_step": step})
                if is_main:
                    print(f"eval step {step} | loss {eval_metrics['eval/loss']:.4f} | "
                          f"perplexity {eval_metrics['eval/perplexity']:.3f} | "
                          f"tokens {eval_metrics['eval/tokens']} | {eval_metrics['time/eval_seconds']:.2f}s", flush=True)
                if local_logs is not None:
                    local_logs.log_eval(step, eval_metrics)
                    local_logs.summary.update(eval_summary)
                    local_logs.save_json("summary.json", local_logs.summary)
                if wandb_run is not None:
                    wandb_run.summary.update(eval_summary)
            if local_logs is not None:
                local_logs.log_metrics(step, metrics)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if save_every > 0 and (step % save_every == 0 or step == planned_steps):
                # All ranks wait so checkpoint upload time is excluded from the next step timing.
                if is_main and checkpoint_callback is not None:
                    checkpoint_callback(step)
                if distributed:
                    dist.barrier()
            total_loss = 0
            step_start = time.perf_counter()
        if step == train_steps:
            break

    elapsed = elapsed_seconds(epoch_start)
    if is_main:
        if batches_processed == len(dataloader):
            print(f"Epoch time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        else:
            print(f"Training time (partial epoch): {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    summary = {"time/train_seconds": elapsed, "train/completed_steps": step,
               "train/epoch_completed": batches_processed == len(dataloader)}
    if batches_processed == len(dataloader):
        summary["time/epoch_seconds"] = elapsed
    if wandb_run is not None:
        wandb_run.summary.update(summary)
    if local_logs is not None:
        local_logs.summary.update(summary)
        local_logs.save_json("summary.json", local_logs.summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen2.5-7B-bnb-4bit")
    parser.add_argument("--dataset_path", type=str, default="data/sft/train.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="Separate validation JSONL with prompt/response fields.")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size per GPU.")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument(
        "--shuffle_document", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32, help="Global effective batch size across all GPUs and accumulation steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--shuffle_dataset", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clipping", type=float, default=1.0)
    parser.add_argument("--wandb_project", type=str, default="cs336-sft")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--hf_repo_id", type=str, default=None, help="Upload to username/repo-name after training.")
    parser.add_argument("--hf_private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="outputs/sft_adapter")
    parser.add_argument("--log_dir", type=str, default=None, help="Log root; default: output_dir/logs. Each run gets a timestamped directory.")
    parser.add_argument("--save_every", type=int, default=100, help="Save adapter checkpoints every N optimizer steps and at the last step; 0 disables periodic checkpoints.")

    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.steps <= 0 or args.gradient_accumulation_steps <= 0 or args.batch_size <= 0:
        parser.error("steps, gradient_accumulation_steps, and batch_size must be positive")
    if args.save_every < 0:
        parser.error("save_every must be nonnegative")
    if args.eval_every <= 0 or args.eval_batch_size <= 0:
        parser.error("eval_every and eval_batch_size must be positive")
    if not Path(args.eval_dataset_path).is_file():
        parser.error(f"Evaluation dataset not found: {args.eval_dataset_path}")
    if not Path(args.dataset_path).is_file():
        parser.error(f"Training dataset not found: {args.dataset_path}")
    if Path(args.eval_dataset_path).samefile(args.dataset_path):
        parser.error("Training and evaluation must use separate files")
    if args.batch_size % (world_size * args.gradient_accumulation_steps):
        parser.error("batch_size must be divisible by world_size * gradient_accumulation_steps")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        parser.error("A CUDA GPU is required for each training process")
    torch.cuda.set_device(local_rank)
    with local_logging(args, world_size) as logs:
        try:
            if world_size > 1:
                dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
            run_training(args, local_rank, world_size, logs)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def run_training(args, local_rank, world_size, local_logs=None):
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    torch.manual_seed(args.seed)
    if is_main:
        print("Checking Hugging Face and W&B connections...", flush=True)
    hf_api = prepare_hf_repo(args.hf_repo_id, args.hf_private) if is_main and args.hf_repo_id else None
    if is_main and args.wandb_mode == "online":
        login_wandb()
    run_context = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={**vars(args), "world_size": world_size,
                "per_device_micro_batch_size": args.batch_size // (world_size * args.gradient_accumulation_steps)},
        mode=args.wandb_mode,
    ) if is_main else nullcontext(None)
    with run_context as run:
        if is_main and args.wandb_mode == "online":
            print(f"W&B run: {run.url}", flush=True)
            if local_logs is not None:
                local_logs.summary["wandb/url"] = run.url
        if dist.is_initialized():
            dist.barrier()
        if is_main:
            print(f"Loading model: {args.model_id}", flush=True)
        tokenizer, model = get_model_and_tokenizer(args.model_id, device_index=local_rank)
        if is_main:
            print(f"Loading dataset: {args.dataset_path}", flush=True)
        dataset = SFTDataset(tokenizer, args.dataset_path, args.seq_length, args.shuffle_document, args.seed)
        if is_main:
            print(f"Loading validation dataset: {args.eval_dataset_path}", flush=True)
        eval_dataset = SFTDataset(tokenizer, args.eval_dataset_path, args.seq_length, False, args.seed)
        if len(eval_dataset) == 0:
            raise ValueError("Validation data is too short to form a sequence; check eval_dataset_path/seq_length.")
        eval_rank = dist.get_rank() if dist.is_initialized() else 0
        # No DistributedSampler padding or dropped tail: each packed sequence is evaluated once.
        eval_shard = Subset(eval_dataset, range(eval_rank, len(eval_dataset), world_size))
        eval_dataloader = DataLoader(eval_shard, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
        if is_main:
            print(f"Validation packed sequences: {len(eval_dataset)}; interval: {args.eval_every} steps", flush=True)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=dist.get_rank(),
            shuffle=args.shuffle_dataset, seed=args.seed, drop_last=True,
        ) if dist.is_initialized() else None
        if sampler is not None:
            sampler.set_epoch(0)
        micro_batch_size = args.batch_size // (world_size * args.gradient_accumulation_steps)
        dataloader = DataLoader(
            dataset, batch_size=micro_batch_size, sampler=sampler,
            shuffle=args.shuffle_dataset if sampler is None else False,
            drop_last=True,
        )
        if len(dataloader) == 0:
            raise ValueError("Dataset is too small for one full microbatch per GPU; reduce batch_size.")
        if is_main:
            available_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
            skipped_samples = len(dataset) - len(dataloader) * micro_batch_size * world_size
            print(f"GPUs: {world_size}; microbatch/GPU: {micro_batch_size}; "
                  f"accumulation: {args.gradient_accumulation_steps}; global batch: {args.batch_size}; "
                  f"steps this epoch: {min(args.steps, available_steps)}; dropped tail sequences: {skipped_samples}", flush=True)
        if dist.is_initialized():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        checkpoint_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

        def checkpoint_callback(step):
            url = save_checkpoint(model, tokenizer, args, step, checkpoint_run_id, hf_api)
            if url is not None:
                run.summary["hf/latest_checkpoint_url"] = url
            if local_logs is not None:
                local_logs.summary["checkpoint/latest_step"] = step
                if url is not None:
                    local_logs.summary["hf/latest_checkpoint_url"] = url
                local_logs.save_json("summary.json", local_logs.summary)

        if is_main:
            print("Starting training...", flush=True)
        sft_train(model, optimizer, dataloader, args.gradient_accumulation_steps, args.clipping, args.steps,
                  wandb_run=run, local_logs=local_logs, save_every=args.save_every,
                  checkpoint_callback=checkpoint_callback if is_main else None,
                  eval_dataloader=eval_dataloader, eval_every=args.eval_every)
        if not is_main:
            return
        unwrapped_model = model.module if isinstance(model, DDP) else model
        print(f"Saving adapter and tokenizer to: {args.output_dir}", flush=True)
        unwrapped_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Adapter and tokenizer saved to: {args.output_dir}", flush=True)
        if hf_api is not None:
            print(f"Uploading to Hugging Face: {args.hf_repo_id}", flush=True)
            try:
                commit = hf_api.upload_folder(
                    repo_id=args.hf_repo_id,
                    repo_type="model",
                    folder_path=args.output_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"],
                    ignore_patterns=["logs/**", "checkpoints/**"],
                    commit_message="Upload trained SFT adapter and tokenizer",
                )
            except Exception:
                print(f"Hugging Face upload failed. Local files remain in {args.output_dir} for retry.", flush=True)
                raise
            run.summary["hf/commit_url"] = commit.commit_url
            if local_logs is not None:
                local_logs.summary["hf/commit_url"] = commit.commit_url
            print(f"Hugging Face upload complete: {commit.commit_url}", flush=True)

if __name__ == "__main__":
    main()
