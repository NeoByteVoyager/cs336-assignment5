"""Contract checks for vLLM requests/export without requiring GPU kernels."""

from types import SimpleNamespace
from types import ModuleType
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file

from cs336_alignment import vllm_utils
from scripts.train_grpo_kaggle import VLLMRollout, cuda_link_environment, generate, inspect_test_samples, parse_args


def test_cuda_link_environment_uses_loaded_driver(tmp_path, monkeypatch):
    driver = tmp_path / "libcuda.so.555.1"
    driver.touch()
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw:
        f"0000-ffff r-xp 0000 00:00 1 {driver}\n" if str(p) == "/proc/self/maps"
        else original_read(p, *a, **kw))
    monkeypatch.setenv("LIBRARY_PATH", "/existing")
    for _ in range(2):
        env = cuda_link_environment(tmp_path)
        assert env["LIBRARY_PATH"] == f"{tmp_path}/cuda_link:/existing"
        assert (tmp_path / "cuda_link/libcuda.so").resolve() == driver


def test_vllm_requests_grouped_independent_samples_and_entropy():
    calls = []

    def complete(prompts, params, batch_size):
        calls.append((prompts, params.copy(), batch_size))
        return [vllm_utils.VLLMCompletion(text=f"{p}:{i}", token_ids=[1], finish_reason="stop",
                                         token_log_probs=[-1.0, -2.0])
                for p in prompts for i in range(params["n"])]

    backend = VLLMRollout.__new__(VLLMRollout)
    backend.server = SimpleNamespace(generate_completions=complete)
    backend.request_seed = 10
    args = SimpleNamespace(prompt="r1_zero", max_new_tokens=512, vllm_request_batch_size=256)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
    responses, lengths, entropy = generate(backend, tokenizer, ["A"] * 8 + ["B"] * 8, args, True)
    assert responses == [f"{p}:{i}" for p in ("A", "B") for i in range(8)]
    assert lengths == [1] * 16 and entropy == 1.5
    assert calls[0][0] == ["A", "B"]
    assert calls[0][1]["n"] == 8
    assert calls[0][1]["seed"] == 10 and backend.request_seed == 26
    assert calls[0][1]["stop"] == ["</answer>"]
    assert calls[0][1]["logprobs"] == 0
    assert calls[0][2] == 256
    generate(backend, tokenizer, ["C", "D"], args)
    assert calls[1][1]["n"] == 1 and calls[1][1]["seed"] == 26


def test_prompt_seeds_disjoint_and_out_of_order_choices_aligned(monkeypatch):
    requests = {}
    def http(method, url, payload, timeout):
        prompt = payload["prompt"][0]
        requests[prompt] = payload
        return {"choices": [dict(index=i, text=f"{prompt}:{i}") for i in reversed(range(8))]}
    monkeypatch.setattr(vllm_utils, "_http_json", http)
    result = vllm_utils.generate_completions("http://localhost", "model", ["A", "B", "C"],
        dict(temperature=1., max_tokens=512, n=8, seed=10, independent_prompt_seeds=True), batch_size=32)
    assert [c.text for c in result] == [f"{p}:{i}" for p in "ABC" for i in range(8)]
    assert [requests[p]["seed"] for p in "ABC"] == [10, 18, 26]
    assert all(p["add_special_tokens"] is False for p in requests.values())


def test_completion_client_forwards_sampling_and_parses_log_probs(monkeypatch):
    requests = []

    def http(method, url, payload, timeout):
        requests.append(payload)
        return dict(choices=[dict(index=0, text="answer", token_ids=[2], finish_reason="stop",
                                  logprobs=dict(token_logprobs=[-0.7]))])

    monkeypatch.setattr(vllm_utils, "_http_json", http)
    completions = vllm_utils.generate_completions("http://localhost", "model", ["p"],
        dict(temperature=1.0, max_tokens=512, n=1, seed=0, top_p=1.0, top_k=-1, logprobs=0))
    assert requests[0]["logprobs"] == 0
    assert requests[0]["top_p"] == 1.0 and requests[0]["top_k"] == -1
    assert completions[0].token_log_probs == [-0.7]


def test_vllm_server_launch_uses_fp16_and_memory_limits(monkeypatch):
    captured = {}

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return object()

    monkeypatch.setattr(vllm_utils.subprocess, "Popen", popen)
    vllm_utils.start_server("model", "127.0.0.1", 8000, 1, 42, "auto", "INFO", 0.75,
                           env_overrides={"LIBRARY_PATH": "/test/cuda_link"},
                           dtype="float16", extra_args=("--max-num-seqs", "32", "--enforce-eager"))
    command = captured["command"]
    assert command[command.index("--dtype") + 1] == "float16"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.75"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["env"]["VLLM_SERVER_DEV_MODE"] == "1"
    assert captured["env"]["LIBRARY_PATH"] == "/test/cuda_link"
    assert "--enforce-eager" in command


def test_vllm_export_saves_half_weights_without_changing_train_model(tiny_train_model, tmp_path):
    backend = VLLMRollout.__new__(VLLMRollout)
    backend.policy = tiny_train_model
    backend.save_pretrained(tmp_path, max_shard_size="1GB")
    tensors = load_file(tmp_path / "model.safetensors")
    assert all(tensor.dtype == torch.float16 for tensor in tensors.values() if tensor.is_floating_point())
    assert all(p.dtype == torch.float32 for p in tiny_train_model.parameters())
    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == "float16"
    assert not list(tmp_path.glob("*.pt"))


def test_vllm_inspection_restores_request_seed(tmp_path):
    backend = VLLMRollout.__new__(VLLMRollout)
    backend.request_seed = 99
    seen = []

    def fake_generate(tokenizer, prompts, args, collect_entropy):
        seen.append(backend.request_seed)
        backend.request_seed += 1
        return ["42"], [1], None

    backend.generate = fake_generate
    score = lambda *args: dict(reward=1.0, format_reward=1.0, answer_reward=1.0)
    for phase in ("before_rl", "after_rl"):
        inspect_test_samples(backend, None, [dict(question="Question", answer="#### 42")],
                             "{question}", score, SimpleNamespace(data_seed=42), tmp_path, phase)
        assert backend.request_seed == 99
    assert seen == [42, 42]


def test_session_defaults_and_budget_validation():
    args = parse_args([])
    assert args.inference_backend == "vllm"
    assert args.max_hours == 11.5
    assert args.vllm_memory_fraction == 0.75
    assert args.skip_completed
    with pytest.raises(SystemExit):
        parse_args(["--max-hours", "12"])


def test_weight_sync_streams_half_parameters_and_invalidates_cache(monkeypatch):
    module_name = "vllm.distributed.weight_transfer.nccl_engine"
    module = ModuleType(module_name)
    sent, requests = {}, []

    def send(iterator, trainer_args):
        sent.update(weights=list(iterator), packed=trainer_args.packed)

    module.NCCLTrainerSendWeightsArgs = lambda **kwargs: SimpleNamespace(**kwargs)
    module.NCCLWeightTransferEngine = SimpleNamespace(trainer_send_weights=send)
    for name in ("vllm", "vllm.distributed", "vllm.distributed.weight_transfer"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)

    def http(method, url, payload=None, timeout=None):
        requests.append((url, payload))
        return {}

    monkeypatch.setattr(vllm_utils, "_http_json", http)
    model = torch.nn.Linear(2, 2)
    vllm_utils.sync_policy_weights(model, "http://localhost", object(), packed=False, transfer_dtype=torch.float16)
    assert not sent["packed"]
    assert all(tensor.dtype == torch.float16 for _, tensor in sent["weights"])
    update = next(payload["update_info"] for url, payload in requests if url.endswith("/update_weights"))
    assert update["dtype_names"] == ["float16", "float16"]
    assert update["packed"] is False
    assert [url.rsplit("/", 1)[1] for url, _ in requests] == ["pause", "update_weights", "reset_prefix_cache", "resume"]
