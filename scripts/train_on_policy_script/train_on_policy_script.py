#!/usr/bin/env python3
"""Two-GPU, full-parameter mixed-precision GRPO using the project's loss components.

See scripts/train_on_policy_script/README_grpo_onpolicy.md. FP16 AMP and FP32 CPU master weights; original implementation untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `python scripts/train_on_policy_script/train_on_policy_script.py` without installing this project.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.onpolicy_grpo.grpo_components import grpo_train_step
from cs336_alignment.onpolicy_grpo.utils import tokenize_prompt_and_output, get_reponse_log_probs
from scripts.train_on_policy_script.amp_train_step import MasterWeightOptimizer, grpo_train_step_amp, cast_model_to_fp16
from scripts.train_on_policy_script.grpo_runtime import (
    TrainingOptimizer, TransformersRollouts, VLLMRollouts, TokenizedResponse, RolloutTokenizer, check_vllm_dependency,
)

DEFAULT_PROMPT = (
    'Solve the following problem step by step. End with your final answer in '
    '\\boxed{{}}.\nQuestion: {question}\nAnswer:'
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-name', default='allenai/OLMo-2-0425-1B')
    p.add_argument('--train-path', default='data/gsm8k/train.jsonl')
    p.add_argument('--val-path', default='data/gsm8k/test.jsonl')
    p.add_argument('--prompt-template', default=str(Path(__file__).resolve().parents[2] / 'cs336_alignment/prompts/r1_zero.prompt'), help='UTF-8 file, Python format with {question}; escape literal braces as {{ }}')
    p.add_argument('--reward', choices=['question_only', 'r1_zero'], default='r1_zero')
    p.add_argument('--output-dir', default='runs/grpo')
    p.add_argument('--resume', help='resume model/step from a trusted checkpoint; optimizer state starts fresh')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    p.add_argument('--amp-init-scale', type=float, default=128.0)
    p.add_argument('--attn-implementation', choices=['sdpa', 'eager'], default='sdpa')
    p.add_argument('--train-gpu', type=int, default=0)
    p.add_argument('--rollout-gpu', type=int, default=1, help='-1: share training GPU; default: second GPU')
    p.add_argument('--rollout-backend', choices=['transformers', 'vllm'], default=None,
                   help='default: vllm on CUDA, transformers for CPU smoke tests')
    p.add_argument('--rollout-batch-size', type=int, default=32, help='vLLM concurrent sequence limit / response batching cap')
    p.add_argument('--optimizer', choices=['adamw_cpu', 'sgd'], default='adamw_cpu')
    p.add_argument('--train-size', type=int, default=6400)
    p.add_argument('--stop-string', default='auto', help='auto: </answer> for r1_zero; empty string disables')
    p.add_argument('--entropy-every', type=int, default=1)
    p.add_argument('--keep-checkpoints', type=int, default=1, help='remove older complete checkpoints BEFORE saving; 0 keeps all')
    p.add_argument('--vllm-port', type=int, default=8000)
    p.add_argument('--vllm-memory-utilization', type=float, default=0.75)
    p.add_argument('--vllm-startup-timeout', type=int, default=600)
    p.add_argument('--train-steps', type=int, default=100, help='total rollout iterations, including resumed steps; gradient overflow may skip an update')
    p.add_argument('--prompts-per-step', type=int, default=1)
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--gradient-accumulation-steps', type=int, default=4)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--learning-rate', type=float, default=1e-5, help='AdamW: betas=(0.9,0.95), weight_decay=0; or zero-momentum SGD')
    p.add_argument('--max-prompt-tokens', type=int, default=256, help='filter long questions; never truncate prompts')
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-size', type=int, default=1024)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--log-samples', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb-project')
    p.add_argument('--wandb-name')
    args = p.parse_args(argv)
    if args.rollout_backend is None:
        args.rollout_backend = 'vllm' if args.device == 'cuda' else 'transformers'
    positive = ['train_steps', 'prompts_per_step', 'gradient_accumulation_steps',
                'max_prompt_tokens', 'max_new_tokens', 'eval_every', 'eval_size', 'save_every',
                'rollout_batch_size', 'entropy_every', 'train_size', 'vllm_startup_timeout']
    if any(getattr(args, key) < 1 for key in positive):
        p.error('step counts, sizes, token limits and intervals must be positive')
    if args.keep_checkpoints < 0 or args.train_gpu < 0 or args.rollout_gpu < -1:
        p.error('invalid GPU index or checkpoint retention')
    if not 0 < args.vllm_memory_utilization < 1:
        p.error('vllm-memory-utilization must be in (0,1)')
    if args.rollout_backend == 'vllm' and (args.device != 'cuda' or args.rollout_gpu in (-1, args.train_gpu)):
        p.error('vLLM requires a separate CUDA rollout GPU')
    if args.stop_string == 'auto':
        args.stop_string = '</answer>' if args.reward == 'r1_zero' else ''
    if args.group_size < 2:
        p.error('group-size must be >= 2 (existing torch.std uses correction=1)')
    if (args.prompts_per_step * args.group_size) % args.gradient_accumulation_steps:
        p.error('prompts-per-step * group-size must be divisible by gradient-accumulation-steps')
    if not (args.temperature > 0 and 0 < args.top_p <= 1 and args.top_k >= 0):
        p.error('require temperature > 0, 0 < top-p <= 1, top-k >= 0')
    if args.amp_init_scale <= 0:
        p.error('amp-init-scale must be positive')
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.log_samples < 0:
        p.error('learning-rate and max-grad-norm must be positive; log-samples must be nonnegative')
    if (args.temperature, args.top_p, args.top_k) != (1.0, 1.0, 0):
        warnings.warn('Sampling warpers change the behavior distribution. The existing train step '
                      'scores raw model log-probabilities without correction. For strict on-policy '
                      'sampling use temperature=1, top-p=1, top-k=0.')
    return args


def load_records(path, template, tokenizer, max_prompt_tokens):
    """GSM8K JSONL or JSON list; preserve prompt/answer correspondence."""
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    rows = json.loads(text) if path.suffix == '.json' else [json.loads(x) for x in text.splitlines() if x.strip()]
    records, skipped = [], 0
    for row in rows:
        prompt = template.format(question=row['question'])
        truth = str(row['answer']).rsplit('####', 1)[-1].strip()
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or not truth:
            raise ValueError(f'Empty prompt or ground truth in {path}')
        if len(ids) > max_prompt_tokens:
            skipped += 1
            continue
        records.append({'prompt': prompt, 'ground_truth': truth})
    if not records:
        raise ValueError(f'No usable rows in {path}; increase --max-prompt-tokens')
    print(f'{path}: kept={len(records)}, filtered_long={skipped}', flush=True)
    return records


def generate_responses(model, tokenizer, prompts, args, *, sample):
    """Small left-padded generation batches on the rollout GPU.

    Preserve sampled IDs, including EOS, alongside decoded display/reward text.
    The training tokenizer adapter reads those IDs without re-tokenizing text.
    """
    model.eval()
    config = deepcopy(model.generation_config)
    # Remove inherited sampling constraints; set all behavior explicitly.
    config.do_sample = sample
    config.num_beams = 1
    config.num_beam_groups = 1
    config.num_return_sequences = 1
    config.max_new_tokens = args.max_new_tokens
    config.min_length = 0
    config.min_new_tokens = None
    config.repetition_penalty = 1.0
    config.no_repeat_ngram_size = 0
    config.forced_bos_token_id = None
    config.forced_eos_token_id = None
    config.suppress_tokens = None
    config.begin_suppress_tokens = None
    config.bad_words_ids = None
    config.stop_strings = [args.stop_string] if args.stop_string else None
    config.temperature = args.temperature if sample else 1.0
    config.top_p = args.top_p if sample else 1.0
    config.top_k = args.top_k if sample else 50
    config.use_cache = True
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    responses = []
    device = next(model.parameters()).device
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16,
        enabled=device.type == 'cuda' and args.precision == 'fp16', cache_enabled=False,
    ):
        for start in range(0, len(prompts), args.rollout_batch_size):
            batch = prompts[start:start + args.rollout_batch_size]
            tokenizer.padding_side = 'left'
            inputs = tokenizer(batch, add_special_tokens=False, padding=True, return_tensors='pt').to(device)
            outputs = model.generate(**inputs, generation_config=config, tokenizer=tokenizer)
            for generated in outputs[:, inputs['input_ids'].shape[1]:].tolist():
                # Remove only post-EOS batch padding; keep the actual EOS action.
                if tokenizer.eos_token_id in generated:
                    generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                response = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                # String-based stopping may append batch padding without an EOS.
                if (tokenizer.eos_token_id not in generated and args.stop_string
                        and args.stop_string in response):
                    while generated and generated[-1] == tokenizer.pad_token_id:
                        generated.pop()
                    response = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                responses.append(TokenizedResponse(response, generated))
            print(f'[generate/{"train" if sample else "val"}] '
                  f'{len(responses)}/{len(prompts)} responses completed', flush=True)
    return responses


def expand_group(records, group_size):
    prompts = [row['prompt'] for row in records for _ in range(group_size)]
    truths = [row['ground_truth'] for row in records for _ in range(group_size)]
    return prompts, truths


def score_responses(responses, truths, reward_fn):
    scores = [reward_fn(response, truth) for response, truth in zip(responses, truths, strict=True)]
    return {key: sum(row[key] for row in scores) / len(scores)
            for key in ('reward', 'format_reward', 'answer_reward')}


class Logger:
    def __init__(self, args, config):
        self.root = Path(args.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run = None
        if args.wandb_project:
            import wandb
            self.run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)

    def write(self, event):
        with (self.root / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if self.run:
            self.run.log(event, step=event['step'])

    def progress(self, step, message):
        line = f'[step {step}] {message}'
        print(line, flush=True)
        with (self.root / 'progress.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def samples(self, kind, step, prompts, responses, truths, count):
        rows = [dict(kind=kind, step=step, prompt=p, response=r, ground_truth=t)
                for p, r, t in list(zip(prompts, responses, truths, strict=True))[:count]]
        with (self.root / 'samples.jsonl').open('a', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        if self.run and rows:
            import wandb
            self.run.log({f'{kind}/samples': wandb.Table(columns=list(rows[0]),
                           data=[list(row.values()) for row in rows])}, step=step)


def make_reward_fn(tokenizer, kind):
    base = question_only_reward_fn if kind == 'question_only' else r1_zero_reward_fn
    def reward(response, truth):
        # Preserve EOS for policy loss; remove terminal special tokens for grading.
        clean = response
        while True:
            previous = clean
            for token in (tokenizer.eos_token, tokenizer.pad_token):
                if token and clean.endswith(token):
                    clean = clean[:-len(token)]
            if clean == previous:
                break
        return base(clean, truth)
    return reward


def response_lengths(tokenizer, responses):
    return [len(r.token_ids) if isinstance(r, TokenizedResponse)
            else len(tokenizer.encode(r, add_special_tokens=False)) for r in responses]


@torch.no_grad()
def token_entropy(model, tokenizer, prompts, responses):
    # Extra no-grad forward: existing train_step returns no entropy metadata.
    # This is the raw policy's response-token entropy BEFORE the update.
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for prompt, response in zip(prompts, responses, strict=True):
        batch = {k: v.to(device) for k, v in
                 tokenize_prompt_and_output([prompt], [response], tokenizer).items()}
        entropy = get_reponse_log_probs(model, batch['input_ids'], batch['labels'],
                                        return_token_entropy=True)['token_entropy']
        mask = batch['response_mask']
        total += entropy[mask].sum().item()
        count += mask.sum().item()
    return total / count


def evaluate(rollouts, tokenizer, records, args, reward_fn, step, logger):
    prompts, truths = expand_group(records[:args.eval_size], 1)
    logger.progress(step, f'Validation started: {len(prompts)} questions')
    devices = list(range(torch.cuda.device_count())) if args.device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        responses = rollouts.generate(prompts, sample=False, step=step)
    scores = score_responses(responses, truths, reward_fn)
    logger.write({'step': step, 'val/accuracy': scores['answer_reward'],
                  'val/reward': scores['reward'], 'val/format_reward': scores['format_reward'],
                  'val/response_length': sum(response_lengths(tokenizer, responses)) / len(responses),
                  'val/count': len(responses)})
    logger.samples('val', step, prompts, responses, truths, args.log_samples)


def checkpoint_plan(model, args):
    root = Path(args.output_dir)
    old = sorted(p for p in root.glob('checkpoint-*') if (p / 'trainer_state.pt').is_file())
    remove = old[:max(0, len(old) - args.keep_checkpoints + 1)] if args.keep_checkpoints else []
    reclaimable = sum(f.stat().st_size for p in remove for f in p.rglob('*') if f.is_file())
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    expected = weights + 128 * 2**20  # model + tokenizer + small RNG/config metadata only
    if shutil.disk_usage(root).free + reclaimable < expected:
        raise OSError(f'Insufficient checkpoint disk space: need about {expected / 2**30:.2f} GiB '
                      '(after configured checkpoint retention). Free space or choose another output directory.')
    return remove


def save_checkpoint(model, tokenizer, sampler, args, config, step):
    root = Path(args.output_dir)
    destination = root / f'checkpoint-{step:06d}'
    temporary = root / f'.checkpoint-{step:06d}.incomplete'
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Checkpoint destination already exists: {destination}')
    # Keep disk use bounded. Only remove this script's COMPLETE checkpoints.
    for path in checkpoint_plan(model, args):
        print(f'Removing older checkpoint before save: {path}', flush=True)
        shutil.rmtree(path)
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    torch.save({'step': step, 'optimizer_saved': False,
                'sampler_rng': sampler.getstate(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else None,
                'config': config}, temporary / 'trainer_state.pt')
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    print(f'Saved {destination}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.device == 'cpu' and args.precision == 'fp16':
        print('CPU smoke test: using FP32; CUDA training defaults to FP16 AMP.', flush=True)
        args.precision = 'fp32'
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu only for a tiny smoke test')
    train_device = f'cuda:{args.train_gpu}' if args.device == 'cuda' else 'cpu'
    rollout_device = (f'cuda:{args.rollout_gpu}' if args.rollout_gpu >= 0 else train_device) if args.device == 'cuda' else 'cpu'
    if args.device == 'cuda':
        required = max(args.train_gpu, args.rollout_gpu)
        if torch.cuda.device_count() <= required:
            raise RuntimeError('Requested GPUs unavailable. Dual-T4 default requires two GPUs; use --rollout-gpu -1 for one GPU')
        torch.cuda.set_device(args.train_gpu)
    if args.rollout_backend == 'vllm':
        check_vllm_dependency()  # Fail before downloads/model allocation when installation is missing.
    print(f'Rollout backend={args.rollout_backend}, concurrency={args.rollout_batch_size}', flush=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = random.Random(args.seed)
    template = Path(args.prompt_template).read_text(encoding='utf-8') if args.prompt_template else DEFAULT_PROMPT
    template.format(question='test')  # Validate braces early.
    config = vars(args).copy()
    config['template_text'] = template
    for key in ('train_path', 'val_path'):
        config[key + '_sha256'] = hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest()
    state = None
    if args.resume:
        state = torch.load(Path(args.resume) / 'trainer_state.pt', map_location='cpu', weights_only=False)
        mutable = {'resume', 'output_dir', 'train_steps', 'eval_every', 'save_every',
                   'rollout_backend', 'rollout_batch_size', 'vllm_memory_utilization', 'precision', 'amp_init_scale', 'attn_implementation', 'log_samples', 'keep_checkpoints', 'vllm_port', 'vllm_startup_timeout', 'wandb_project', 'wandb_name', 'train_path', 'val_path', 'prompt_template'}
        changed = [key for key in config if key not in mutable and config[key] != state['config'].get(key)]
        if changed:
            raise ValueError(f'Resume configuration differs: {changed}')
        if state['step'] >= args.train_steps:
            raise ValueError('--train-steps must exceed the checkpoint step')
    source = args.resume or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer needs an EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = load_records(args.train_path, template, tokenizer, args.max_prompt_tokens)
    val = load_records(args.val_path, template, tokenizer, args.max_prompt_tokens)
    if min(len(train), args.train_size) < args.prompts_per_step:
        raise ValueError('Not enough training questions for prompts-per-step')
    train = random.Random(args.seed).sample(train, min(args.train_size, len(train)))
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation=args.attn_implementation,
    ).to(train_device)
    if args.max_prompt_tokens + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Prompt + response token limits exceed model context length')
    # Checkpointing needs train mode; nonzero dropout would mismatch rollout policy.
    if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()) or any(
        isinstance(value, (int, float)) and value > 0
        for key, value in model.config.to_dict().items() if 'dropout' in key
    ):
        raise ValueError('This simple version requires zero dropout (OLMo 2 default)')
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # Clone original FP32 weights before quantizing the GPU model to FP16.
    optimizer = MasterWeightOptimizer(model.parameters(), args.learning_rate, args.optimizer)
    if args.precision == 'fp16':
        cast_model_to_fp16(model)
    scaler = torch.amp.GradScaler(
        args.device, init_scale=args.amp_init_scale, enabled=args.precision == 'fp16',
    )
    optimizer.zero_grad(set_to_none=True)
    start = 0
    if state:
        print('Resuming model weights and step; optimizer state is reset (not restored); AMP loss scale restarts.', flush=True)
        sampler.setstate(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start = state['step']
    logger = Logger(args, config)
    checkpoint_plan(model, args)  # Fail before expensive rollouts if saving cannot fit.
    (logger.root / f'config-from-step-{start}.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    element_bytes = 2 if args.precision == 'fp16' else 4
    print(f'{args.precision.upper()} model parameters+gradients lower bound: '
          f'{sum(p.numel() for p in model.parameters()) * element_bytes * 2 / 2**30:.2f} GiB; '
          f'FP32 CPU master weights, attention={args.attn_implementation}, AMP={scaler.is_enabled()}. '
          'Activations/logits/workspaces need additional memory.', flush=True)
    if args.optimizer == 'adamw_cpu':
        print(f'CPU master parameters + Adam state lower bound: '
              f'{sum(p.numel() for p in model.parameters()) * 12 / 2**30:.2f} GiB; '
              'plus up to one full FP32 gradient buffer during the CPU update '
              f'({sum(p.numel() for p in model.parameters()) * 4 / 2**30:.2f} GiB) and temporary tensors.', flush=True)
    reward_fn = make_reward_fn(tokenizer, args.reward)
    runtime = TransformersRollouts if args.rollout_backend == 'transformers' else VLLMRollouts
    devices = list(range(torch.cuda.device_count())) if args.device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        rollouts = runtime(model, source, rollout_device, tokenizer, args, generate_responses)
    try:
        run_training(model, tokenizer, optimizer, sampler, train, val, args, config,
                     reward_fn, start, logger, rollouts, scaler)
    finally:
        rollouts.close()
        if logger.run:
            logger.run.finish()


def run_training(model, tokenizer, optimizer, sampler, train, val, args, config,
                 reward_fn, start, logger, rollouts, scaler):
    training_tokenizer = RolloutTokenizer(tokenizer)
    logger.progress(start, 'Synchronizing rollout model for initial validation')
    rollouts.sync()
    evaluate(rollouts, tokenizer, val, args, reward_fn, start, logger)
    for step in range(start + 1, args.train_steps + 1):
        began = time.monotonic()
        if args.device == 'cuda':
            for gpu in sorted({args.train_gpu, args.rollout_gpu} - {-1}):
                torch.cuda.reset_peak_memory_stats(gpu)
        selected = sampler.sample(train, args.prompts_per_step)
        prompts, truths = expand_group(selected, args.group_size)
        logger.progress(step, f'Start {step}/{args.train_steps}: syncing weights, then generating {len(prompts)} responses')
        rollouts.sync()
        responses = rollouts.generate(prompts, sample=True, step=step)
        scores = score_responses(responses, truths, reward_fn)
        logger.progress(step, f'Rollout completed: reward={scores["reward"]:.4f}, format_reward={scores["format_reward"]:.4f}')
        logger.progress(step, f'Updating policy: {args.gradient_accumulation_steps} microbatches, '
                        f'precision={args.precision}, optimizer={args.optimizer}')
        # One loss-scaled update per fresh rollout; overflow discards this rollout.
        loss, metadata = grpo_train_step_amp(
            model=model, tokenizer=training_tokenizer, optimizer=optimizer, scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm, reward_fn=reward_fn,
            repeated_prompts=prompts, rollout_responses=responses,
            repeated_ground_truths=truths, group_size=args.group_size,
            precision=args.precision, log_entropy=step % args.entropy_every == 0,
            progress=lambda message: logger.progress(step, message),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss; stop without saving this update')
        event = {'step': step, 'train/loss': loss.item(), 'train/reward': scores['reward'],
                 'train/format_reward': scores['format_reward'],
                 'train/grad_norm': metadata['grad_norm'],
                 'train/amp_scale': metadata['amp_scale'],
                 'train/optimizer_step_skipped': metadata['optimizer_step_skipped'],
                 'train/response_length': sum(response_lengths(tokenizer, responses)) / len(responses),
                 'train/seconds': time.monotonic() - began}
        if 'token_entropy' in metadata:
            event['train/token_entropy'] = metadata['token_entropy']
        if args.device == 'cuda':
            for gpu in sorted({args.train_gpu, args.rollout_gpu} - {-1}):
                event[f'gpu{gpu}/peak_allocated_gib'] = torch.cuda.max_memory_allocated(gpu) / 2**30
        logger.write(event)
        logger.samples('train', step, prompts, responses, truths, args.log_samples)
        if step % args.eval_every == 0 or step == args.train_steps:
            logger.progress(step, 'Syncing current policy weights for validation')
            rollouts.sync()
            evaluate(rollouts, tokenizer, val, args, reward_fn, step, logger)
        if step % args.save_every == 0 or step == args.train_steps:
            logger.progress(step, 'Saving model/tokenizer checkpoint (without optimizer state)')
            save_checkpoint(model, tokenizer, sampler, args, config, step)


if __name__ == '__main__':
    main()
