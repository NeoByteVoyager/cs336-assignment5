"""Execution helpers; the GRPO loss remains in cs336_alignment/onpolicy_grpo."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


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
                source, torch_dtype=torch.float32, attn_implementation='eager').to(device)
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


def build_vllm_command(args, source):
    """Pinned to vLLM 0.19.1. FP32 also keeps weight-transfer dtypes identical."""
    return [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(source),
            '--host', '127.0.0.1', '--port', str(args.vllm_port),
            '--dtype', 'float32', '--enforce-eager', '--attention-backend', 'TRITON_ATTN',
            '--tensor-parallel-size', '1', '--seed', str(args.seed),
            '--max-model-len', str(args.max_prompt_tokens + args.max_new_tokens),
            '--max-num-seqs', str(args.rollout_batch_size),
            '--max-num-batched-tokens', str(max(1024, args.max_prompt_tokens + args.max_new_tokens)),
            '--gpu-memory-utilization', str(args.vllm_memory_utilization),
            '--generation-config', 'vllm',
            '--weight-transfer-config', json.dumps({'backend': 'nccl'})]


class VLLMRollouts:
    """Reuse existing NCCL synchronization, with a T4-compatible server launch.

    Opt-in: startup, Triton kernels and NCCL must be verified on the target Kaggle
    driver. No fallback to stale weights or the initial checkpoint is permitted.
    """
    def __init__(self, policy, source, device, tokenizer, args, generate_fn=None):
        from importlib.metadata import version
        from cs336_alignment.vllm_utils import VLLMServer, wait_for_server
        if version('vllm') != '0.19.1':
            raise RuntimeError('The optional vLLM backend requires vllm==0.19.1')
        self.policy, self.tokenizer, self.args = policy, tokenizer, args
        self.server = VLLMServer(str(source), port=args.vllm_port,
                                 gpu=torch.device(device).index, seed=args.seed,
                                 launch_server=False)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(torch.device(device).index),
                   VLLM_SERVER_DEV_MODE='1', VLLM_LOGGING_LEVEL='INFO')
        self.log = (Path(args.output_dir) / 'vllm-server.log').open('a')
        self.server.process = subprocess.Popen(build_vllm_command(args, source), env=env,
                                               stdout=self.log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
        atexit.register(self.close)
        try:
            wait_for_server(self.server.base_url, self.server.process, args.vllm_startup_timeout)
            self.server.init_weight_sync(str(next(policy.parameters()).device))
        except BaseException:
            self.close()
            raise

    def sync(self):
        self.server.sync_policy_weights(self.policy)

    def generate(self, prompts, *, sample, step):
        from cs336_alignment.vllm_utils import _http_json
        responses = []
        for start in range(0, len(prompts), self.args.rollout_batch_size):
            # Token prompts match the existing tokenizer helper (no implicit BOS).
            payload = {'model': self.server.model_id,
                       'prompt': [self.tokenizer.encode(p, add_special_tokens=False)
                                  for p in prompts[start:start + self.args.rollout_batch_size]],
                       'temperature': self.args.temperature if sample else 0.0,
                       'top_p': self.args.top_p if sample else 1.0,
                       'top_k': self.args.top_k if sample and self.args.top_k else -1,
                       'max_tokens': self.args.max_new_tokens, 'n': 1,
                       'seed': self.args.seed + step * 100000 + start,
                       'return_token_ids': True, 'skip_special_tokens': False}
            if self.args.stop_string:
                payload.update(stop=[self.args.stop_string], include_stop_str_in_output=True)
            result = _http_json('POST', self.server.base_url + '/v1/completions', payload, timeout=3600)
            choices = sorted(result['choices'], key=lambda row: row['index'])
            if len(choices) != len(payload['prompt']):
                raise RuntimeError('vLLM returned an unexpected number of completions')
            for row in choices:
                ids = row.get('token_ids')
                if not ids:
                    raise RuntimeError('vLLM did not return nonempty token_ids')
                text = self.tokenizer.decode(ids, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
                if self.tokenizer.encode(text, add_special_tokens=False) != ids:
                    raise RuntimeError('vLLM output tokens do not round-trip through the existing text-only train step')
                responses.append(text)
            print(f'[generate/{"train" if sample else "val"}] '
                  f'{len(responses)}/{len(prompts)} responses completed', flush=True)
        return responses

    def close(self):
        self.server.stop()
        self.log.close()
