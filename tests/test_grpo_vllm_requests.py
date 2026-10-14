"""Offline protocol checks; no vLLM install, GPU or monkey patch required."""
import pytest
from .test_grpo_onpolicy_script import runner, tiny
from scripts.train_on_policy_script.grpo_runtime import (
    vllm_generation_requests, decode_vllm_choices, build_vllm_command,
)


def test_cuda_defaults_to_vllm_and_cpu_defaults_to_transformers():
    assert runner.parse_args([]).rollout_backend == 'vllm'
    assert runner.parse_args([]).rollout_batch_size == 32
    assert runner.parse_args(['--device', 'cpu']).rollout_backend == 'transformers'
    with pytest.raises(SystemExit):
        runner.parse_args(['--device', 'cpu', '--rollout-backend', 'vllm'])


def test_grouped_requests_use_parallel_sampling_and_preserve_order(tiny):
    _, tokenizer, *_ = tiny
    args = runner.parse_args(['--group-size', '2', '--gradient-accumulation-steps', '2', '--rollout-batch-size', '4'])
    prompts = ['Q 2 A', 'Q 2 A', 'Q 3 A', 'Q 3 A', 'Q A', 'Q A']
    requests = list(vllm_generation_requests(tokenizer, prompts, args, sample=True, step=7, model_id='policy'))
    assert [len(p['prompt']) for p in requests] == [2, 1]
    assert all(p['n'] == 2 and p['return_token_ids'] and not p['add_special_tokens'] for p in requests)
    assert requests[0]['prompt'] == [tokenizer.encode(p, add_special_tokens=False) for p in prompts[::2][:2]]
    assert requests[0]['seed'] != requests[1]['seed']
    assert requests[0]['temperature'] == 1.0 and requests[0]['top_p'] == 1.0 and requests[0]['top_k'] == -1
    # Simulate an out-of-order API response; client must restore prompt-major order.
    rows = [{'index': i, 'token_ids': [token]} for i, token in enumerate([7, 8, 5, 6])]
    result = decode_vllm_choices(tokenizer, {'choices': rows[::-1]}, 4)
    assert [r.token_ids for r in result] == [(7,), (8,), (5,), (6,)]
    assert all(isinstance(r, str) for r in result)


def test_validation_is_greedy_one_completion_per_question(tiny):
    _, tokenizer, *_ = tiny
    args = runner.parse_args(['--group-size', '8', '--rollout-batch-size', '32'])
    request, = vllm_generation_requests(tokenizer, ['Q 2 A', 'Q 3 A'], args,
                                        sample=False, step=0, model_id='policy')
    assert request['n'] == 1 and len(request['prompt']) == 2
    assert request['temperature'] == 0.0
    assert request['stop'] == ['</answer>'] and request['include_stop_str_in_output']


def test_invalid_groups_and_incomplete_choices_fail(tiny):
    _, tokenizer, *_ = tiny
    args = runner.parse_args(['--group-size', '2', '--gradient-accumulation-steps', '2'])
    with pytest.raises(ValueError, match='grouped consecutively'):
        list(vllm_generation_requests(tokenizer, ['Q 2 A', 'Q 3 A'], args,
                                      sample=True, step=1, model_id='policy'))
    with pytest.raises(ValueError, match='complete consecutive groups'):
        list(vllm_generation_requests(tokenizer, ['Q'], args, sample=True, step=1, model_id='policy'))
    with pytest.raises(RuntimeError, match='indexes'):
        decode_vllm_choices(tokenizer, {'choices': [{'index': 0, 'token_ids': [7]}] * 2}, 2)
    with pytest.raises(RuntimeError, match='token_ids'):
        decode_vllm_choices(tokenizer, {'choices': [{'index': 0, 'token_ids': []}]}, 1)
    command = build_vllm_command(args, 'policy')
    assert '--enable-prefix-caching' in command
    assert command[command.index('--max-num-seqs') + 1] == '32'
