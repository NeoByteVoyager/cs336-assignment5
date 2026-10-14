"""Offline precision checks; CPU FP16 exercises scaling, not CUDA kernel support."""
import math

import pytest
import torch

from .test_grpo_onpolicy_script import tiny, runner
from scripts.train_on_policy_script.amp_train_step import MasterWeightOptimizer, grpo_train_step_amp


def inputs():
    return dict(repeated_prompts=['Q 2 A'] * 4,
                rollout_responses=['\\boxed{2}', '\\boxed{3}', '\\boxed{2}', '\\boxed{3}'],
                repeated_ground_truths=['2'] * 4, group_size=2,
                reward_fn=runner.question_only_reward_fn)


@pytest.mark.parametrize('accumulation', [1, 4])
def test_fp32_matches_original_step(tiny, accumulation):
    model, tokenizer, *_ = tiny
    reference = type(model)(model.config)
    reference.load_state_dict(model.state_dict())
    opt = MasterWeightOptimizer(model.parameters(), 0.1, 'sgd')
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    loss, metadata = grpo_train_step_amp(model, tokenizer, opt, scaler, accumulation, 1e-3,
                                        precision='fp32', **inputs())
    old_loss, _ = runner.grpo_train_step(reference, tokenizer,
                                       torch.optim.SGD(reference.parameters(), lr=0.1),
                                       accumulation, 1e-3, **inputs())
    torch.testing.assert_close(loss, old_loss, atol=1e-6, rtol=1e-5)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-5)
    assert not metadata['optimizer_step_skipped']
    assert metadata['grad_norm'] > 1e-3  # verifies clipping against the reference step
    assert math.isfinite(metadata['token_entropy'])
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in opt.masters)


def test_fp16_master_weights_update_and_scale(tiny):
    model, tokenizer, *_ = tiny
    opt = MasterWeightOptimizer(model.parameters(), 0.01)
    initial = [p.detach().clone() for p in opt.masters]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    model.half()
    scaler = torch.amp.GradScaler('cpu', init_scale=8.0)
    loss, metadata = grpo_train_step_amp(model, tokenizer, opt, scaler, 4, 1.0, **inputs())
    assert torch.isfinite(loss)
    assert not metadata['optimizer_step_skipped']
    assert metadata['grad_norm'] > 0
    assert any(not torch.equal(p, old) for p, old in zip(opt.masters, initial))
    for p, master in zip(model.parameters(), opt.masters):
        assert p.dtype == torch.float16 and master.dtype == torch.float32
        torch.testing.assert_close(p, master.half(), rtol=0, atol=0)
        assert p.grad is None and master.grad is None
    assert all(float(state['step']) == 1 for state in opt.inner.state.values())
    assert all(state['exp_avg'].dtype == torch.float32 for state in opt.inner.state.values())


def test_fp16_overflow_skips_entire_update_and_recovers(tiny):
    model, tokenizer, *_ = tiny
    opt = MasterWeightOptimizer(model.parameters(), 0.01)
    model.half()
    initial_model = [p.detach().clone() for p in model.parameters()]
    initial_master = [p.detach().clone() for p in opt.masters]
    scaler = torch.amp.GradScaler('cpu', init_scale=1e20)
    loss, metadata = grpo_train_step_amp(model, tokenizer, opt, scaler, 2, 1.0, **inputs())
    assert torch.isfinite(loss)
    assert metadata['optimizer_step_skipped'] and metadata['grad_norm'] is None
    assert scaler.get_scale() < 1e20
    assert not opt.inner.state
    for p, old in zip(model.parameters(), initial_model):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
    for p, old in zip(opt.masters, initial_master):
        torch.testing.assert_close(p, old, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in opt.masters)
    scaler.update(new_scale=8.0)
    _, metadata = grpo_train_step_amp(model, tokenizer, opt, scaler, 2, 1.0, **inputs())
    assert not metadata['optimizer_step_skipped']
    assert all(float(state['step']) == 1 for state in opt.inner.state.values())


def test_fp16_checkpoint_contains_model_not_optimizer(tiny, tmp_path):
    import random
    from safetensors import safe_open
    model, tokenizer, *_ = tiny
    opt = MasterWeightOptimizer(model.parameters(), 0.01)
    model.half()
    scaler = torch.amp.GradScaler('cpu', init_scale=8.0)
    grpo_train_step_amp(model, tokenizer, opt, scaler, 4, 1.0, **inputs())
    assert opt.inner.state  # Adam state exists in RAM, so this is a meaningful disk check.
    args = runner.parse_args(['--device', 'cpu', '--output-dir', str(tmp_path)])
    runner.save_checkpoint(model, tokenizer, random.Random(1), args, vars(args), 1)
    checkpoint = tmp_path / 'checkpoint-000001'
    state = torch.load(checkpoint / 'trainer_state.pt', weights_only=False)
    assert state['optimizer_saved'] is False and 'optimizer' not in state and 'scaler' not in state
    assert (checkpoint / 'trainer_state.pt').stat().st_size < 65536
    with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
        assert saved.keys()
        assert all(saved.get_tensor(k).dtype == torch.float16 for k in saved.keys())
    loaded = type(model).from_pretrained(checkpoint, dtype='auto')
    for p, q in zip(model.parameters(), loaded.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_cast_keeps_rotary_buffers_in_fp32(tiny):
    from scripts.train_on_policy_script.amp_train_step import cast_model_to_fp16
    model, *_ = tiny
    before = {k: v.clone() for k, v in model.named_buffers() if v.is_floating_point()}
    assert before  # OLMo rotary buffers are present.
    cast_model_to_fp16(model)
    assert all(p.dtype == torch.float16 for p in model.parameters())
    for name, buffer in model.named_buffers():
        if name in before:
            assert buffer.dtype == torch.float32
            torch.testing.assert_close(buffer, before[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU is unavailable locally')
@pytest.mark.parametrize('overflow', [False, True])
def test_cuda_scaler_with_cpu_master_optimizer(tiny, overflow):
    from scripts.train_on_policy_script.amp_train_step import cast_model_to_fp16
    model, tokenizer, *_ = tiny
    model.cuda()
    opt = MasterWeightOptimizer(model.parameters(), 0.01)
    cast_model_to_fp16(model)
    before = [p.detach().clone() for p in model.parameters()]
    scaler = torch.amp.GradScaler('cuda', init_scale=1e20 if overflow else 8.0)
    loss, metadata = grpo_train_step_amp(model, tokenizer, opt, scaler, 4, 1.0, **inputs())
    assert torch.isfinite(loss)
    assert metadata['optimizer_step_skipped'] == overflow
    assert all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in opt.masters)
    if overflow:
        assert not opt.inner.state
        for p, old in zip(model.parameters(), before):
            torch.testing.assert_close(p, old, rtol=0, atol=0)
    else:
        assert opt.inner.state
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))


def test_sampled_ids_padding_and_eos_mask_match_original(tiny):
    model, tokenizer, *_ = tiny
    tokenizer.pad_token = tokenizer.eos_token
    adapter = runner.RolloutTokenizer(tokenizer)
    data = inputs()
    data['repeated_prompts'] = ['Q [EOS] A'] * 4
    # Deliberately noncanonical action sequences with different lengths. Text is
    # for the reward; both train steps must score these attached sampled IDs.
    data['rollout_responses'] = [
        runner.TokenizedResponse('\\boxed{2}', [5, 6, 7, 2]),
        runner.TokenizedResponse('\\boxed{3}', [8, 2]),
        runner.TokenizedResponse('\\boxed{2}', [7, 2]),
        runner.TokenizedResponse('\\boxed{3}', [6, 5, 8, 2]),
    ]
    reference = type(model)(model.config)
    reference.load_state_dict(model.state_dict())
    opt = MasterWeightOptimizer(model.parameters(), 0.01, 'sgd')
    loss, _ = grpo_train_step_amp(model, adapter, opt, torch.amp.GradScaler('cpu', enabled=False),
                                  4, 1.0, precision='fp32', **data)
    expected, _ = runner.grpo_train_step(reference, adapter,
                                        torch.optim.SGD(reference.parameters(), lr=0.01),
                                        4, 1.0, **data)
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-5)
