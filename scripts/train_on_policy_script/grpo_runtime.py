"""Execution helpers; the GRPO loss remains in cs336_alignment/onpolicy_grpo."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

import torch


class TokenizedResponse(str):
    """Decoded display/reward text carrying the original sampled token IDs.

    Metadata belongs to this response object, so identical text from different
    token sequences remains distinguishable. JSON serializes the ordinary text.
    """
    def __new__(cls, text, token_ids):
        obj = super().__new__(cls, text)
        obj.token_ids = tuple(token_ids)
        if not obj.token_ids:
            raise ValueError('Generation returned no response tokens')
        return obj

    def __getnewargs__(self):
        return str(self), self.token_ids


class RolloutTokenizer:
    """Explicit adapter for the existing text-only train-step interface.

    For generated responses use their sampled IDs; delegate ordinary prompt
    encoding and tokenizer attributes. Does not mutate the HF tokenizer or
    replace any existing function.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, TokenizedResponse):
            if add_special_tokens or kwargs:
                raise ValueError('Sampled responses require encode(add_special_tokens=False) without other options')
            return list(text.token_ids)
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)


class TrainingOptimizer(torch.optim.Optimizer):
    """Explicit optimizer adapter: log/clip once, optionally run PyTorch AdamW on CPU.

    The existing train step must receive max_grad_norm=None. Its ordinary step()
    call reaches this adapter AFTER all microbatch backward calls. No patch/hooks.
    """
    def __init__(self, params, lr, max_grad_norm, kind='adamw_cpu'):
        params = list(params)
        super().__init__(params, {'lr': lr})
        self.kind = kind
        self.max_grad_norm = max_grad_norm
        self.grad_norm = float('nan')
        if kind == 'adamw_cpu':
            self.masters = [torch.nn.Parameter(p.detach().to('cpu', copy=True)) for p in params]
            self.inner = torch.optim.AdamW(self.masters, lr=lr, betas=(0.9, 0.95),
                                           weight_decay=0.0, foreach=False)
        elif kind == 'sgd':
            self.masters = None
            self.inner = torch.optim.SGD(params, lr=lr, momentum=0, foreach=False)
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not supported')
        params = self.param_groups[0]['params']
        self.grad_norm = float(torch.nn.utils.clip_grad_norm_(
            params, self.max_grad_norm, error_if_nonfinite=True, foreach=False))
        if self.masters is None:
            return self.inner.step()
        # Only one parameter-sized gradient copy is resident on CPU at a time.
        # AdamW's per-parameter step counters still advance exactly once/update.
        for param, master in zip(params, self.masters, strict=True):
            if param.grad is None:
                continue
            master.grad = param.grad.detach().to('cpu', copy=True)
            self.inner.step()
            param.copy_(master)
            master.grad = None

    def state_dict(self):
        return {'kind': self.kind, 'inner': self.inner.state_dict()}

    def load_state_dict(self, state_dict):
        if state_dict['kind'] != self.kind:
            raise ValueError('Cannot change optimizer when resuming')
        self.inner.load_state_dict(state_dict['inner'])


class TransformersRollouts:
    def __init__(self, policy, source, device, tokenizer, args, generate_fn):
        from transformers import AutoModelForCausalLM
        self.policy = policy
        self.tokenizer = tokenizer
        self.args = args
        self.generate_fn = generate_fn
        if torch.device(device) == next(policy.parameters()).device:
            self.model = policy
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                source, dtype=torch.float16 if args.precision == 'fp16' and args.device == 'cuda' else torch.float32,
                attn_implementation=args.attn_implementation).to(device)
            self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def sync(self):
        if self.model is self.policy:
            return
        # Copy all parameters and buffers (including non-persistent model buffers).
        for target, source in [(dict(self.model.named_parameters()), dict(self.policy.named_parameters())),
                               (dict(self.model.named_buffers()), dict(self.policy.named_buffers()))]:
            if target.keys() != source.keys():
                raise ValueError('Rollout model structure differs from policy')
            for name in target:
                target[name].copy_(source[name])

    def generate(self, prompts, *, sample, step):
        return self.generate_fn(self.model, self.tokenizer, prompts, self.args, sample=sample)

    def close(self):
        pass


def check_vllm_dependency():
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version('vllm')
    except PackageNotFoundError as exc:
        raise RuntimeError(
            'vLLM is the default CUDA rollout backend. Install it first with: '
            'python -m pip install --no-cache-dir -r '
            'scripts/train_on_policy_script/requirements-grpo-vllm.txt '
            '(or explicitly select --rollout-backend transformers)'
        ) from exc
    if installed != '0.19.1':
        raise RuntimeError(f'This weight-sync backend requires vllm==0.19.1; found {installed}')


def vllm_generation_requests(tokenizer, prompts, args, *, sample, step, model_id):
    """One parent request per question, n children with different sampling seeds.

    Public Completion API indexes choices in prompt-major, then child order.
    Grouping n=1 duplicate parents under one seed would duplicate their RNG.
    """
    n = args.group_size if sample else 1
    if len(prompts) % n:
        raise ValueError('Training prompt batch must contain complete consecutive groups')
    unique_prompts = []
    for start in range(0, len(prompts), n):
        group = prompts[start:start + n]
        if any(prompt != group[0] for prompt in group):
            raise ValueError('Responses for the same question must be grouped consecutively')
        unique_prompts.append(group[0])
    question_batch_size = max(1, args.rollout_batch_size // n)
    for start in range(0, len(unique_prompts), question_batch_size):
        payload = {'model': model_id,
                   'prompt': [tokenizer.encode(p, add_special_tokens=False)
                              for p in unique_prompts[start:start + question_batch_size]],
                   'add_special_tokens': False,
                   'temperature': args.temperature if sample else 0.0,
                   'top_p': args.top_p if sample else 1.0,
                   'top_k': args.top_k if sample and args.top_k else -1,
                   'min_p': 0.0, 'repetition_penalty': 1.0,
                   'max_tokens': args.max_new_tokens, 'n': n,
                   'seed': args.seed + step * 100000 + start * n,
                   'return_token_ids': True, 'skip_special_tokens': False}
        if args.stop_string:
            payload.update(stop=[args.stop_string], include_stop_str_in_output=True)
        yield payload


def decode_vllm_choices(tokenizer, result, expected):
    choices = sorted(result['choices'], key=lambda row: row['index'])
    if [row['index'] for row in choices] != list(range(expected)):
        raise RuntimeError('vLLM returned missing, duplicate or unexpected completion indexes')
    responses = []
    for row in choices:
        ids = row.get('token_ids')
        if not ids:
            raise RuntimeError('vLLM did not return nonempty token_ids')
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        responses.append(TokenizedResponse(text, ids))
    return responses


def build_vllm_command(args, source):
    """Pinned to vLLM 0.19.1; inference and policy weight dtypes must match."""
    return [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(source),
            '--host', '127.0.0.1', '--port', str(args.vllm_port),
            '--dtype', 'float16' if args.precision == 'fp16' else 'float32', '--enforce-eager', '--attention-backend', 'TRITON_ATTN',
            '--tensor-parallel-size', '1', '--seed', str(args.seed),
            '--max-model-len', str(args.max_prompt_tokens + args.max_new_tokens),
            '--max-num-seqs', str(args.rollout_batch_size),
            '--max-num-batched-tokens', str(max(2048, args.max_prompt_tokens + args.max_new_tokens)),
            '--enable-prefix-caching',
            '--gpu-memory-utilization', str(args.vllm_memory_utilization),
            '--generation-config', 'vllm',
            '--weight-transfer-config', json.dumps({'backend': 'nccl'})]


class VLLMRollouts:
    """Reuse existing NCCL synchronization, with a T4-compatible server launch.

    Startup, Triton kernels and NCCL must be verified on the target Kaggle
    driver. No fallback to stale weights or the initial checkpoint is permitted.
    """
    def __init__(self, policy, source, device, tokenizer, args, generate_fn=None):
        from cs336_alignment.vllm_utils import VLLMServer, wait_for_server
        check_vllm_dependency()
        # Refuse a busy port rather than accidentally syncing another server.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', args.vllm_port))
        self.policy, self.tokenizer, self.args = policy, tokenizer, args
        self.server = VLLMServer(str(source), port=args.vllm_port,
                                 gpu=torch.device(device).index, seed=args.seed,
                                 launch_server=False)
        env = os.environ.copy()
        visible = env.get('CUDA_VISIBLE_DEVICES')
        physical_gpu = (visible.split(',')[torch.device(device).index].strip()
                        if visible else str(torch.device(device).index))
        env.update(CUDA_VISIBLE_DEVICES=physical_gpu,
                   VLLM_SERVER_DEV_MODE='1', VLLM_LOGGING_LEVEL='INFO')
        self.log_path = Path(args.output_dir) / 'vllm-server.log'
        self.log = self.log_path.open('a')
        print(f'Starting vLLM: precision={args.precision}, device={device}, '
              f'max-num-seqs={args.rollout_batch_size}; server log: {self.log_path}', flush=True)
        self.server.process = subprocess.Popen(build_vllm_command(args, source), env=env,
                                               stdout=self.log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
        atexit.register(self.close)
        try:
            wait_for_server(self.server.base_url, self.server.process, args.vllm_startup_timeout)
            self.server.init_weight_sync(str(next(policy.parameters()).device))
        except Exception as exc:
            self.close()
            raise RuntimeError(f'vLLM startup/weight-sync initialization failed; inspect {self.log_path}: {exc}') from exc
        except BaseException:
            self.close()
            raise

    def sync(self):
        self.server.sync_policy_weights(self.policy)

    def generate(self, prompts, *, sample, step):
        from cs336_alignment.vllm_utils import _http_json
        responses = []
        for payload in vllm_generation_requests(self.tokenizer, prompts, self.args,
                                               sample=sample, step=step, model_id=self.server.model_id):
            result = _http_json('POST', self.server.base_url + '/v1/completions', payload, timeout=3600)
            responses.extend(decode_vllm_choices(self.tokenizer, result, len(payload['prompt']) * payload['n']))
            print(f'[vllm/generate/{"train" if sample else "val"}] '
                  f'{len(responses)}/{len(prompts)} responses completed', flush=True)
        return responses

    def close(self):
        self.server.stop()
        self.log.close()
