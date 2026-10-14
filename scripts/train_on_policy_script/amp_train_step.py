"""FP16 GRPO execution with FP32 master weights; original assignment files untouched."""
from __future__ import annotations

import torch
from cs336_alignment.onpolicy_grpo.grpo_components import (
    compute_rollout_rewards, compute_group_normalized_reward,
    compute_policy_gradient_loss, aggregate_loss_across_microbatch,
)
from cs336_alignment.onpolicy_grpo.utils import tokenize_prompt_and_output


def cast_model_to_fp16(model):
    """Keep floating buffers (e.g. rotary frequencies) in FP32 while casting weights."""
    buffers = [(module, name, value.float()) for module in model.modules()
               for name, value in module.named_buffers(recurse=False) if value.is_floating_point()]
    model.half()
    for module, name, value in buffers:
        setattr(module, name, value)
    return model


class MasterWeightOptimizer(torch.optim.Optimizer):
    """Public GradScaler-compatible optimizer over CPU FP32 master parameters.

    GPU model gradients may be FP16. Copy/cast them to CPU FP32 BEFORE calling
    scaler.unscale_(self); GradScaler never unscales FP16 parameter gradients.
    PyTorch AdamW handles the master parameters, then updated weights are copied
    back to the model. No optimizer state is included in model.save_pretrained.
    """
    def __init__(self, model_parameters, lr, kind='adamw_cpu'):
        self.model_parameters = list(model_parameters)
        self.masters = [torch.nn.Parameter(p.detach().to(device='cpu', dtype=torch.float32, copy=True))
                        for p in self.model_parameters]
        super().__init__(self.masters, {'lr': lr})
        self.kind = kind
        self.grad_norm = None
        if kind == 'adamw_cpu':
            self.inner = torch.optim.AdamW(self.masters, lr=lr, betas=(0.9, 0.95),
                                           weight_decay=0.0, foreach=False)
        elif kind == 'sgd':
            self.inner = torch.optim.SGD(self.masters, lr=lr, momentum=0, foreach=False)
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def prepare_gradients(self):
        for model_param, master in zip(self.model_parameters, self.masters, strict=True):
            master.grad = (None if model_param.grad is None else
                           model_param.grad.detach().to(device='cpu', dtype=torch.float32, copy=True))
            model_param.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are not supported')
        self.inner.step()
        for model_param, master in zip(self.model_parameters, self.masters, strict=True):
            model_param.copy_(master)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for param in self.model_parameters:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def state_dict(self):
        return {'kind': self.kind, 'inner': self.inner.state_dict()}

    def load_state_dict(self, state):
        if state['kind'] != self.kind:
            raise ValueError('Cannot change optimizer kind')
        self.inner.load_state_dict(state['inner'])


def _response_entropy(log_probs, mask):
    """Entropy of the raw policy, token weighted; bounded no-grad temporaries."""
    total = 0.0
    with torch.no_grad():
        for row, row_mask in zip(log_probs.detach(), mask, strict=True):
            for start in range(0, row.shape[0], 32):
                block = row[start:start + 32]
                entropy = -(block.exp() * block).sum(dim=-1)
                total += entropy[row_mask[start:start + 32]].sum().item()
    return total


def grpo_train_step_amp(
    model, tokenizer, optimizer, scaler, gradient_accumulation_steps, max_grad_norm,
    reward_fn, repeated_prompts, rollout_responses, repeated_ground_truths, group_size,
    *, precision='fp16', log_entropy=True, progress=None,
):
    """One fresh rollout -> at most one update, using existing GRPO loss components.

    FP16 model forward/autograd + FP32 log-softmax/loss. Gradients accumulate at
    one loss scale over the WHOLE rollout. Transfer to FP32 masters, unscale once,
    check/clip once, step once, update scale once, and clear all gradients.
    Gradient overflow skips the optimizer update and discards this rollout.
    """
    count = len(rollout_responses)
    if not count or not (count == len(repeated_prompts) == len(repeated_ground_truths)):
        raise ValueError('Nonempty aligned prompts/responses/truths required')
    if group_size < 2 or count % group_size or gradient_accumulation_steps < 1 or count % gradient_accumulation_steps:
        raise ValueError('Group/accumulation sizes must divide the rollout batch; group_size >= 2')
    if precision not in ('fp16', 'fp32'):
        raise ValueError(precision)
    if precision == 'fp16' and not scaler.is_enabled():
        raise ValueError('FP16 training requires an enabled GradScaler')
    device = next(model.parameters()).device
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    lengths = [len(tokenizer.encode(p, add_special_tokens=False))
               + len(tokenizer.encode(r, add_special_tokens=False)) - 1
               for p, r in zip(repeated_prompts, rollout_responses, strict=True)]
    raw_rewards, reward_metrics = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, _ = compute_group_normalized_reward(raw_rewards, group_size, 'mean', 1e-6, 'std')
    microbatch_size = count // gradient_accumulation_steps
    batch_loss, entropy_sum, entropy_tokens = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for index, start in enumerate(range(0, count, microbatch_size), 1):
        end = start + microbatch_size
        width = max(lengths[start:end])
        inputs = batch['input_ids'][start:end, :width].to(device)
        labels = batch['labels'][start:end, :width].to(device)
        mask = batch['response_mask'][start:end, :width].to(device)
        if not mask.any(dim=-1).all():
            raise ValueError('Each response must contain at least one sampled token')
        attention_mask = torch.arange(inputs.shape[1], device=device)[None, :] < torch.tensor(
            lengths[start:end], device=device)[:, None]
        # No BF16. Disable the weight-cast cache to bound temporary memory.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=precision == 'fp16', cache_enabled=False):
            logits = model(inputs, attention_mask=attention_mask, use_cache=False).logits
        if not torch.isfinite(logits).all():
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite forward logits: update aborted. Loss scaling cannot fix FP16 activation overflow.')
        # Normalization and the policy-gradient loss remain FP32.
        all_log_probs = torch.log_softmax(logits.float(), dim=-1)
        del logits
        selected = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        losses, _ = compute_policy_gradient_loss(
            advantages[start:end].to(device), selected, importance_reweighting_method='none')
        loss = aggregate_loss_across_microbatch(losses, mask, loss_normalization='sequence')
        loss = loss * (microbatch_size / count)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite forward loss: update aborted. This is not fixed by loss scaling; inspect FP16 activations.')
        batch_loss += loss.detach().item()
        if log_entropy:
            entropy_sum += _response_entropy(all_log_probs, mask)
            entropy_tokens += mask.sum().item()
        scaler.scale(loss).backward()
        del loss, losses, selected, all_log_probs
        if progress and (index % max(1, gradient_accumulation_steps // 8) == 0 or index == gradient_accumulation_steps):
            progress(f'Backward {index}/{gradient_accumulation_steps} microbatches completed')
    if progress:
        progress('Transferring gradients to FP32 CPU master weights; unscaling and clipping')
    optimizer.prepare_gradients()
    scaler.unscale_(optimizer)
    norms = [p.grad.detach().norm(2) for p in optimizer.masters if p.grad is not None]
    norm = torch.stack(norms).norm(2) if norms else torch.tensor(0.0)
    finite = bool(torch.isfinite(norm))
    previous_scale = scaler.get_scale()
    if finite:
        optimizer.grad_norm = norm.item()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(optimizer.masters, max_grad_norm, error_if_nonfinite=True, foreach=False)
        if progress:
            progress('Applying FP32 CPU optimizer update and synchronizing model weights')
        scaler.step(optimizer)
        scaler.update()
    elif scaler.is_enabled():
        # Also covers a finite gradient tensor whose FP32 total-norm reduction
        # overflows. Do not let scaler.step() apply such an update.
        optimizer.grad_norm = None
        bad_gradient = any(not torch.isfinite(p.grad).all().item()
                           for p in optimizer.masters if p.grad is not None)
        if bad_gradient:
            scaler.step(optimizer)  # GradScaler detects inf/NaN and does NOT call optimizer.step.
            scaler.update()  # Normal backoff also resets the finite-step growth counter.
        else:
            scaler.update(new_scale=previous_scale * scaler.get_backoff_factor())
        if progress:
            progress('Gradient overflow: optimizer update skipped; loss scale reduced')
    else:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('Nonfinite FP32 gradient norm; optimizer update aborted')
    optimizer.zero_grad(set_to_none=True)
    metadata = {**reward_metrics, 'grad_norm': optimizer.grad_norm,
                'amp_scale': scaler.get_scale(), 'optimizer_step_skipped': not finite}
    if log_entropy:
        metadata['token_entropy'] = entropy_sum / entropy_tokens
    return torch.tensor(batch_loss), metadata
