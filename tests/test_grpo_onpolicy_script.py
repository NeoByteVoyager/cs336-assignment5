"""CPU integration checks, offline: real HF generate + existing GRPO train step."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

SPEC = importlib.util.spec_from_file_location(
    'onpolicy_script', Path(__file__).resolve().parents[1] / 'scripts/train_on_policy_script/train_on_policy_script.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def tiny(tmp_path):
    # Every token round-trips, including EOS, with no external downloads.
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'Q': 3, 'A': 4,
             '2': 5, '3': 6, '\\boxed{2}': 7, '\\boxed{3}': 8}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                        pad_token='[PAD]', eos_token='[EOS]')
    torch.manual_seed(123)
    model = Olmo2ForCausalLM(Olmo2Config(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=1, eos_token_id=2,
        attention_dropout=0.0, tie_word_embeddings=False))
    source = tmp_path / 'model'
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    data = tmp_path / 'data.jsonl'
    data.write_text('\n'.join(json.dumps({'question': q, 'answer': f'work #### {q}'})
                              for q in ['2', '3']))
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('Q {question} A')
    return model, tokenizer, source, data, prompt


def test_existing_step_updates_and_accumulates(tiny):
    model, tokenizer, *_ = tiny
    responses = ['\\boxed{2}', '\\boxed{3}', '\\boxed{3}', '\\boxed{2}']
    prompts, truths = runner.expand_group(
        [{'prompt': 'Q 2 A', 'ground_truth': '2'}, {'prompt': 'Q 3 A', 'ground_truth': '3'}], 2)
    assert truths == ['2', '2', '3', '3']
    assert prompts == ['Q 2 A', 'Q 2 A', 'Q 3 A', 'Q 3 A']
    other = Olmo2ForCausalLM(model.config)
    other.load_state_dict(model.state_dict())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for current, accumulation in [(model, 4), (other, 1)]:
        optimizer = torch.optim.SGD(current.parameters(), lr=0.01)
        loss, _ = runner.grpo_train_step(
            current, tokenizer, optimizer, accumulation, 1.0,
            runner.question_only_reward_fn, prompts, responses, truths, 2)
        assert torch.isfinite(loss)
        assert all(p.grad is None for p in current.parameters())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, other.state_dict()[k], atol=1e-7, rtol=1e-5)


def test_generate_train_save_resume(tiny, tmp_path, capsys):
    _, _, source, data, prompt = tiny
    common = ['--model-name', str(source), '--train-path', str(data), '--val-path', str(data),
              '--prompt-template', str(prompt), '--reward', 'question_only', '--device', 'cpu', '--group-size', '2',
              '--gradient-accumulation-steps', '2', '--max-prompt-tokens', '8',
              '--max-new-tokens', '4', '--eval-size', '2', '--eval-every', '1',
              '--save-every', '1', '--learning-rate', '0.01']
    uninterrupted = tmp_path / 'full'
    split = tmp_path / 'split'
    runner.main(common + ['--output-dir', str(uninterrupted), '--train-steps', '2'])
    runner.main(common + ['--output-dir', str(split), '--train-steps', '1'])
    runner.main(common + ['--output-dir', str(split), '--train-steps', '2',
                          '--resume', str(split / 'checkpoint-000001')])
    # Model checkpoint loads, but a fresh Adam optimizer intentionally changes the trajectory.
    resumed = Olmo2ForCausalLM.from_pretrained(split / 'checkpoint-000002')
    assert all(torch.isfinite(p).all() for p in resumed.parameters())
    state_path = split / 'checkpoint-000002/trainer_state.pt'
    state = torch.load(state_path, weights_only=False)
    assert 'optimizer' not in state and state['optimizer_saved'] is False
    assert state_path.stat().st_size < 65536
    assert state['step'] == 2
    assert 'optimizer state is reset' in capsys.readouterr().out
    progress = (split / 'progress.log').read_text()
    assert 'Generating' in progress or 'generating' in progress
    assert 'Updating policy' in progress and 'Validation started' in progress
    metrics = [json.loads(x) for x in (split / 'metrics.jsonl').read_text().splitlines()]
    assert [x['step'] for x in metrics if 'train/loss' in x] == [1, 2]
    assert all('val/accuracy' in x for x in metrics if 'val/count' in x)
    assert (split / 'samples.jsonl').read_text()
    assert not (split / 'checkpoint-000001').exists()
    assert all('train/grad_norm' in x and 'train/token_entropy' in x for x in metrics if 'train/loss' in x)
    assert all('val/response_length' in x and 'val/reward' in x for x in metrics if 'val/count' in x)
    with pytest.raises(ValueError, match='configuration differs'):
        runner.main(common + ['--output-dir', str(split), '--train-steps', '3',
                              '--temperature', '0.9', '--resume', str(split / 'checkpoint-000002')])


def test_bad_cli():
    for args in [['--group-size', '1'], ['--gradient-accumulation-steps', '3']]:
        with pytest.raises(SystemExit):
            runner.parse_args(args)


def test_cpu_adamw_matches_pytorch_and_restores():
    torch.manual_seed(12)
    params = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(2))]
    expected = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opt = runner.TrainingOptimizer(params, 1e-3, 0.5)
    ref = torch.optim.AdamW(expected, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    for _ in range(3):
        for p, q in zip(params, expected):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        norm = torch.nn.utils.clip_grad_norm_(expected, 0.5, foreach=False)
        ref.step()
        opt.step()
        assert opt.grad_norm == float(norm)
        for p, q in zip(params, expected):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        opt.zero_grad()
        ref.zero_grad()
    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = runner.TrainingOptimizer(restored_params, 1e-3, 0.5)
    import copy
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p, q in zip(params, restored_params):
        p.grad = torch.ones_like(p)
        q.grad = p.grad.clone()
    opt.step()
    restored.step()
    for p, q in zip(params, restored_params):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_reward_strips_eos_for_grading_only(tiny):
    _, tokenizer, *_ = tiny
    reward = runner.make_reward_fn(tokenizer, 'r1_zero')
    response = 'work</think> <answer>2</answer>' + tokenizer.eos_token
    assert reward(response, '2')['reward'] == 1.0
    assert response.endswith(tokenizer.eos_token)


def test_vllm_command_uses_fp32_t4_backend():
    from scripts.train_on_policy_script.grpo_runtime import build_vllm_command
    args = runner.parse_args(['--rollout-backend', 'vllm'])
    cmd = build_vllm_command(args, 'model')
    assert cmd[cmd.index('--dtype') + 1] == 'float32'
    assert cmd[cmd.index('--attention-backend') + 1] == 'TRITON_ATTN'
    assert '--weight-transfer-config' in cmd
    assert '--enforce-eager' in cmd


def test_transformers_replica_sync(tiny):
    model, tokenizer, source, *_ = tiny
    args = runner.parse_args(['--device', 'cpu', '--reward', 'question_only'])
    # cpu:0 requests a distinct copy while keeping this test offline and CPU-only.
    replica = runner.TransformersRollouts(model, source, 'cpu:0', tokenizer, args,
                                         runner.generate_responses)
    assert replica.model is not model
    with torch.no_grad():
        next(model.parameters()).add_(0.125)
    assert not torch.equal(next(model.parameters()), next(replica.model.parameters()))
    replica.sync()
    for p, q in zip(model.parameters(), replica.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_kaggle_launcher_cpu_smoke(tiny, tmp_path):
    import os
    import subprocess
    import sys
    _, _, source, data, prompt = tiny
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / 'launched'
    env = dict(os.environ, GRPO_OUTPUT_DIR=str(out),
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    result = subprocess.run(
        ['bash', 'scripts/train_on_policy_script/run_kaggle_grpo.sh', 'smoke', '--device', 'cpu',
         '--model-name', str(source), '--train-path', str(data), '--val-path', str(data),
         '--prompt-template', str(prompt), '--reward', 'question_only',
         '--max-prompt-tokens', '8', '--max-new-tokens', '4'],
        cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout
    assert (out / 'checkpoint-000002/trainer_state.pt').is_file()
    console = (out / 'console.log').read_text()
    assert 'Saved' in console
    assert '[generate/train]' in console and '[generate/val]' in console
    assert 'train/loss' in console and 'train/reward' in console
    assert 'Updating policy' in console
    state = torch.load(out / 'checkpoint-000002/trainer_state.pt', weights_only=False)
    assert 'optimizer' not in state
    assert len([x for x in (out / 'metrics.jsonl').read_text().splitlines()
                if 'train/loss' in x]) == 2
