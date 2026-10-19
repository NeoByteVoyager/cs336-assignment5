"""CPU checks for the Kaggle script's optimizer and memory-saving batching."""

from copy import deepcopy
from types import SimpleNamespace
import csv
import json
import math
import sys

import pytest
import torch

from scripts.train_grpo_kaggle import (
    CPUAdamW,
    RolloutEntropy,
    VARIANTS,
    generate,
    console_log,
    inspect_test_samples,
    parse_args,
    save_best_model,
    train_batch,
    write_comparison,
)
from cs336_alignment.onpolicy_grpo.grpo_components import grpo_train_step


def test_cpu_adamw_matches_torch():
    generator = torch.Generator().manual_seed(42)
    actual = [torch.nn.Parameter(torch.randn(5, 7, generator=generator)) for _ in range(2)]
    expected = [torch.nn.Parameter(p.detach().clone()) for p in actual]
    options = dict(lr=0.01, betas=(0.9, 0.95), weight_decay=0.1)
    offload = CPUAdamW(actual, **options)
    reference = torch.optim.AdamW(expected, foreach=False, **options)
    for step in range(5):
        for p, q in zip(actual, expected):
            gradient = torch.randn(p.shape, generator=generator)
            p.grad = gradient.clone()
            q.grad = gradient.clone()
        # Verify that absent gradients leave parameters and moments unchanged.
        if step == 2:
            actual[1].grad = expected[1].grad = None
        offload.step()
        reference.step()
        for p, q in zip(actual, expected):
            torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)
    assert all(value.device.type == "cpu" for state in offload.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))


def reward(response, truth):
    value = float(response == "world")
    return dict(reward=value, format_reward=1.0, answer_reward=value)


@pytest.mark.parametrize("variant", VARIANTS)
def test_kaggle_microbatch_update_matches_existing_train_step(tiny_train_model, tokenizer, variant):
    reference = deepcopy(tiny_train_model)
    # The second group has zero advantages in all mean-baseline variants;
    # lengths differ, so incorrect per-microbatch normalization is observable.
    prompts = ["Hello", "Hello", "This is", "This is"]
    responses = ["world", "a test", "another test", "test"]
    truths = ["42"] * 4
    args = SimpleNamespace(group_size=2, max_new_tokens=8, microbatch_size=1,
                           gradient_accumulation_steps=2, max_grad_norm=1.0)
    optimizer = CPUAdamW(tiny_train_model.parameters(), lr=1e-3)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, betas=(0.9, 0.95),
                                            weight_decay=0.0, foreach=False)
    metadata = train_batch(tiny_train_model, tokenizer, optimizer,
                           torch.amp.GradScaler("cuda", enabled=False),
                           prompts, responses, truths, reward, args, VARIANTS[variant])
    loss, _ = grpo_train_step(
        reference, tokenizer, reference_optimizer, 2, 1.0, reward,
        prompts, responses, truths, group_size=2,
        normalization_constant=len(prompts) * args.max_new_tokens, **VARIANTS[variant],
    )
    assert metadata["loss"] == pytest.approx(loss.item(), abs=1e-7)
    for p, q in zip(tiny_train_model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
    assert all(p.grad is None for p in tiny_train_model.parameters())


def test_all_zero_advantages_skip_forward_and_optimizer(tiny_train_model, tokenizer, monkeypatch):
    def unexpected_forward(*args, **kwargs):
        pytest.fail("Zero-advantage batch must not run the model")
    monkeypatch.setattr(tiny_train_model, "forward", unexpected_forward)
    optimizer = CPUAdamW(tiny_train_model.parameters())
    metadata = train_batch(
        tiny_train_model, tokenizer, optimizer, torch.amp.GradScaler("cuda", enabled=False),
        ["Hello", "Hello"], ["test", "a test"], ["42", "42"], reward,
        SimpleNamespace(group_size=2), VARIANTS["maxrl"],
    )
    assert metadata["active_sequences"] == 0
    assert metadata["optimizer_step"] == 0
    assert not optimizer.state


@pytest.mark.parametrize("variant", VARIANTS)
def test_update_matches_independent_full_batch_formula(tiny_train_model, tokenizer, variant):
    from cs336_alignment.onpolicy_grpo.utils import tokenize_prompt_and_output
    reference = deepcopy(tiny_train_model)
    prompts = ["Hello"] * 4
    responses = ["world", "a test", "world", "world"]
    truths = ["42"] * 4
    args = SimpleNamespace(group_size=2, max_new_tokens=8, microbatch_size=1,
                           gradient_accumulation_steps=2, max_grad_norm=1e6)
    opt = torch.optim.SGD(tiny_train_model.parameters(), lr=0.1)
    train_batch(tiny_train_model, tokenizer, opt, torch.amp.GradScaler("cuda", enabled=False),
                prompts, responses, truths, reward, args, VARIANTS[variant])
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    logits = reference(batch["input_ids"]).logits
    logp = logits.log_softmax(-1).gather(-1, batch["labels"].unsqueeze(-1)).squeeze(-1)
    rewards = torch.tensor([[1., 0.], [1., 1.]])
    advantage = rewards if variant == "rft" else rewards - rewards.mean(-1, keepdim=True)
    if variant in ("grpo", "grpo_constant"):
        advantage = advantage / (rewards.std(-1, keepdim=True) + 1e-6)
    elif variant == "maxrl":
        advantage = advantage / (rewards.mean(-1, keepdim=True) + 1e-6)
    per_sequence = (logp * batch["response_mask"]).sum(-1)
    if variant == "grpo":
        objective = -(advantage.flatten() * per_sequence / batch["response_mask"].sum(-1)).mean()
    else:
        objective = -(advantage.flatten() * per_sequence).sum() / (4 * 8)
    objective.backward()
    with torch.no_grad():
        for actual, initial in zip(tiny_train_model.parameters(), reference.parameters()):
            expected = initial - 0.1 * initial.grad
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_assignment_hyperparameters():
    args = parse_args([])
    assert (args.steps, args.train_examples, args.val_examples) == (200, 6400, 1024)
    assert (args.rollout_batch_size, args.group_size, args.gradient_accumulation_steps) == (256, 8, 32)
    assert (args.learning_rate, args.max_new_tokens, args.max_grad_norm) == (1e-5, 512, 1.0)
    assert args.seeds == [0]
    assert args.save_model
    assert args.inspect_examples == 100


def test_validation_is_per_answer_and_saves_all_samples(monkeypatch):
    import scripts.train_grpo_kaggle as script
    rows = [dict(question=f"Question {i}", answer=f"work #### {i}") for i in range(12)]
    def generated(model, tokenizer, prompts, args):
        assert prompts == [f"Question {i}" for i in range(12)]
        return [str(i) if i < 3 else "wrong" for i in range(12)], [1] * 12, None
    monkeypatch.setattr(script, "generate", generated)
    def grader(response, truth):
        return dict(reward=float(response == truth), format_reward=1.)
    metrics, samples = script.evaluate(None, None, rows, "{question}", grader, None)
    assert metrics["val_reward"] == 0.25
    assert len(samples) == 12
    assert sum(s["reward"] for s in samples) == 3


def test_multiple_runs_keep_one_best_model():
    assert parse_args(["--variant", "all"]).save_model
    assert parse_args(["--seeds", "0", "1"]).save_model
    assert not parse_args(["--variant", "all", "--no-save-model"]).save_model


def test_best_model_replaces_weights_only_for_improvement(tmp_path):
    class Artifact:
        def __init__(self, filename):
            self.filename = filename
            self.calls = 0

        def save_pretrained(self, directory, **kwargs):
            self.calls += 1
            directory.mkdir(exist_ok=True)
            (directory / self.filename).write_text(str(self.calls))

    model, tokenizer = Artifact("model.safetensors"), Artifact("tokenizer.json")
    for score, expected in [(0.4, True), (0.3, False), (0.4, False), (0.6, True)]:
        candidate = dict(variant="grpo", seed=0, step=10, val_reward=score)
        assert save_best_model(tmp_path, model, tokenizer, candidate) is expected
    assert model.calls == tokenizer.calls == 2
    assert json.loads((tmp_path / "best_model.json").read_text())["val_reward"] == 0.6
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_dir()) == ["best_model"]
    assert sorted(path.name for path in (tmp_path / "best_model").iterdir()) == ["model.safetensors", "tokenizer.json"]


def test_console_output_is_saved_and_still_printed(tmp_path, capsys):
    path = tmp_path / "console.log"
    with console_log(path):
        print("Question and response")
        print("Generation progress", file=sys.stderr)
    captured = capsys.readouterr()
    assert "Question and response" in captured.out
    assert "Generation progress" in captured.err
    assert "Question and response" in path.read_text()
    assert "Generation progress" in path.read_text()


def test_comparison_uses_final_rewards_and_excludes_incomplete_runs(tmp_path, capsys):
    runs = []
    for variant, seed, final, last_step in [("grpo", 0, 0.2, 20), ("grpo", 1, 0.4, 20),
                                           ("grpo", 2, 1.0, 10), ("rft", 0, 0.5, 20)]:
        run_dir = tmp_path / f"{variant}_{seed}"
        run_dir.mkdir()
        config = dict(variant=variant, seed=seed, prompt="r1_zero", steps=20, _run_dir=str(run_dir))
        rows = [dict(step=0, val_reward=0.1, val_format_reward=0.5),
                dict(step=5, val_reward=0.9, val_format_reward=1.0),
                dict(step=last_step, val_reward=final, val_format_reward=1.0)]
        (run_dir / "test_inspection_summary.json").write_text(json.dumps(dict(
            before_rl=dict(mean_reward=0.1), after_rl=dict(mean_reward=0.3), reward_change=0.2)))
        runs.append((config, rows))
    write_comparison(tmp_path, runs)
    with (tmp_path / "comparison.csv").open() as handle:
        compared = list(csv.DictReader(handle))
    assert compared[0]["variant"] == "rft"
    grpo = compared[1]
    assert int(grpo["n_seeds"]) == 2
    assert float(grpo["mean_final_reward"]) == pytest.approx(0.3)
    assert float(grpo["std_final_reward"]) == pytest.approx(math.sqrt(0.02))
    assert float(grpo["mean_inspect_change"]) == pytest.approx(0.2)
    with (tmp_path / "summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 4
    assert summaries[0]["best_val_reward"] == "0.9"
    assert "r1_zero/grpo" in (tmp_path / "comparison.txt").read_text()
    assert "r1_zero/grpo" in capsys.readouterr().out


def test_streaming_entropy_excludes_finished_sequences(tokenizer):
    recorder = RolloutEntropy(tokenizer, 1, 2, "cpu", None)
    scores = torch.zeros(2, len(tokenizer))
    first = torch.tensor([[3], [3]])
    assert recorder(first, scores) is scores
    recorder(torch.tensor([[3, tokenizer.eos_token_id], [3, 4]]), scores)
    assert recorder.count.item() == 3
    assert recorder.total.item() == pytest.approx(3 * torch.log(torch.tensor(float(len(tokenizer)))).item())


def test_generation_runs_on_tiny_model(tiny_train_model, tokenizer):
    tokenizer.padding_side = "left"
    tiny_train_model.eval()
    responses, lengths, entropy = generate(
        tiny_train_model, tokenizer, ["Hello", "This is"],
        SimpleNamespace(prompt="question_only", max_new_tokens=2, generate_batch_size=2),
        collect_entropy=True,
    )
    assert len(responses) == len(lengths) == 2
    assert entropy is not None and entropy > 0


def test_before_after_inspection_logs_same_100_questions_and_preserves_rng(tmp_path, monkeypatch, capsys):
    rows = [dict(question=f"Question {index}", answer="Reasoning #### 42") for index in range(100)]
    calls = []

    def fake_generate(model, tokenizer, prompts, args):
        calls.append((list(prompts), torch.rand(3)))
        return ["world"] * len(prompts), [1] * len(prompts), None

    monkeypatch.setattr("scripts.train_grpo_kaggle.generate", fake_generate)
    model = torch.nn.Linear(1, 1)
    args = SimpleNamespace(data_seed=42)
    for phase in ("before_rl", "after_rl"):
        rng = torch.get_rng_state().clone()
        metrics = inspect_test_samples(model, None, rows, "Solve: {question}", reward, args, tmp_path, phase)
        assert torch.equal(torch.get_rng_state(), rng)
        assert metrics["mean_reward"] == 1.0
        records = [json.loads(line) for line in (tmp_path / f"{phase}_test.jsonl").read_text().splitlines()]
        assert len(records) == 100
        assert [record["question"] for record in records] == [row["question"] for row in rows]
        assert records[-1]["sample_id"] == 100
        assert records[0]["response"] == "world"
        assert records[0]["ground_truth"] == "42"
        # Simulate random draws made during RL between the two inspections.
        torch.rand(7)
    assert calls[0][0] == calls[1][0]
    torch.testing.assert_close(calls[0][1], calls[1][1])
    output = capsys.readouterr().out
    assert "[before_rl] Test 100/100" in output
    assert "[after_rl] Test 100/100" in output
    assert "Question 99" in output and "world" in output
