from __future__ import annotations

import csv
import json
import argparse
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
from typing import TYPE_CHECKING
from cs336_alignment.sft.parse_mmlu_response import parse_mmlu_response

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def load_model(
    model_id: str,
    batch_size: int = 8,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    from vllm import LLM

    print("Loading vLLM model: ", model_id, flush=True)
    model = LLM(
        model=model_id,
        quantization="bitsandbytes",
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_seqs=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        generation_config="vllm",
    )
    print("vLLM model ready", flush=True)
    return model

def load_mmlu_template():
    with open(
        "cs336_alignment/prompts_safety/mmlu_zero_shot.prompt",
        encoding="utf-8",
    ) as f:
        mmlu_zero_shot_template = f.read()

    with open(
        "cs336_alignment/prompts_safety/zero_shot_system_prompt.prompt",
        encoding="utf-8"
    ) as f:
        zero_shot_system_template = f.read()

    return mmlu_zero_shot_template, zero_shot_system_template

def get_templated_mmlu_prompt(
    mmlu_zero_shot_template: str,
    zero_shot_system_template: str,
    subject: str,
    question: str,
    options: list[str]
) -> str:
    mmlu_zero_shot_prompt = mmlu_zero_shot_template.format(
        subject=subject,
        question=question,
        options=options,
    )
    zero_shot_system_prompt = zero_shot_system_template.format(
        instruction=mmlu_zero_shot_prompt
    )
    return zero_shot_system_prompt

def get_responses(
    model: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[str]:
    outputs = model.generate(prompts, sampling_params, use_tqdm=True)
    return [output.outputs[0].text for output in outputs]

def evaluate(
    model: LLM,
    data: str,
    output_dir: str = "outputs/mmlu_baseline",
    batch_size: int = 8,
    worker_rank: int = 0,
    num_workers: int = 1,
):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 <= worker_rank < num_workers:
        raise ValueError("worker_rank must be in [0, num_workers)")
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["# Query:"],
    )
    if data == "mmlu":
        # load template
        mmlu_zero_shot_template, zero_shot_system_template = load_mmlu_template()
        total_correct = 0
        total = 0

        subject_metrics = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        data_dir = Path("data/mmlu/test")
        with (output_path / "predictions.jsonl").open(
            "w", encoding="utf-8", buffering=1
        ) as predictions_file:
            for csv_path in sorted(data_dir.glob("*_test.csv")):
                # record the test theme
                subject = csv_path.stem.removesuffix("_test").replace("_", " ")
                # test every theme
                correct = 0
                subject_total = 0
                with open(csv_path, encoding="utf-8") as f:
                    # Each worker evaluates different rows of every subject.
                    rows = list(enumerate(csv.reader(f)))[worker_rank::num_workers]
                if not rows:
                    continue
                for batch_start in range(0, len(rows), batch_size):
                    batch = rows[batch_start:batch_start + batch_size]
                    prompts = [
                        get_templated_mmlu_prompt(
                            mmlu_zero_shot_template,
                            zero_shot_system_template,
                            subject,
                            line[0],
                            line[1:5],
                        )
                        for _, line in batch
                    ]
                    print(
                        f"[worker {worker_rank}] Generating {subject}: "
                        f"{batch_start + 1}-{batch_start + len(batch)}/{len(rows)}",
                        flush=True,
                    )
                    responses = get_responses(model, prompts, sampling_params)
                    for (example_index, line), response in zip(batch, responses, strict=True):
                        question = line[0]
                        options = line[1:5]
                        answer = line[5]
                        model_ans = parse_mmlu_response(mmlu_example=None, model_output=response)
                        print("question: ", question)
                        print("model_ans: ", model_ans)
                        print("real_ans: ", answer)
                        is_correct = model_ans == answer
                        if is_correct:
                            correct += 1
                        subject_total += 1
                        record = {
                            "subject": subject,
                            "example_index": example_index,
                            "question": question,
                            "options": options,
                            "answer": answer,
                            "response": response,
                            "prediction": model_ans,
                            "is_correct": is_correct,
                        }
                        predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                print("subject acc: ", correct / subject_total)
                subject_metrics[subject] = {
                    "correct": correct,
                    "total": subject_total,
                    "accuracy": correct / subject_total,
                }
                total_correct += correct
                total += subject_total
        acc = total_correct / total if total else 0.0
        print("total acc: ", acc)
        metrics = {
            "correct": total_correct,
            "total": total,
            "accuracy": acc,
            "subjects": subject_metrics,
        }
        with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("results saved to: ", output_path)
        return metrics


def run_worker(gpu_id: str, rank: int, num_workers: int, args):
    # Set visibility before importing vLLM or initializing CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    output_dir = Path(args.output_dir)
    if num_workers > 1:
        output_dir = output_dir / f"worker_{rank}"
    print(f"[worker {rank}] GPU {gpu_id}; output: {output_dir}", flush=True)
    model = load_model(
        args.model_id,
        args.batch_size,
        args.max_model_len,
        args.gpu_memory_utilization,
    )
    evaluate(model, args.data, str(output_dir), args.batch_size, rank, num_workers)


def merge_results(output_dir: str, num_workers: int):
    output_path = Path(output_dir)
    subjects = {}
    records = []
    for rank in range(num_workers):
        worker_dir = output_path / f"worker_{rank}"
        with (worker_dir / "metrics.json").open(encoding="utf-8") as f:
            worker_metrics = json.load(f)
        for subject, counts in worker_metrics["subjects"].items():
            combined = subjects.setdefault(subject, {"correct": 0, "total": 0})
            combined["correct"] += counts["correct"]
            combined["total"] += counts["total"]
        with (worker_dir / "predictions.jsonl").open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)

    for counts in subjects.values():
        counts["accuracy"] = counts["correct"] / counts["total"]
    correct = sum(counts["correct"] for counts in subjects.values())
    total = sum(counts["total"] for counts in subjects.values())
    metrics = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "subjects": subjects,
    }
    records.sort(key=lambda record: (record["subject"], record["example_index"]))
    with (output_path / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Merged {total} examples; total acc: {metrics['accuracy']}", flush=True)
    print("results saved to: ", output_path, flush=True)
    return metrics


def run_parallel(args):
    num_workers = len(args.gpu_ids)
    if num_workers == 1:
        return run_worker(args.gpu_ids[0], 0, 1, args)

    context = mp.get_context("spawn")
    processes = [
        context.Process(target=run_worker, args=(gpu_id, rank, num_workers, args))
        for rank, gpu_id in enumerate(args.gpu_ids)
    ]
    try:
        for process in processes:
            process.start()
        pending = set(processes)
        while pending:
            ready = wait([process.sentinel for process in pending])
            for process in list(pending):
                if process.sentinel in ready:
                    process.join()
                    pending.remove(process)
                    if process.exitcode != 0:
                        raise RuntimeError(
                            f"Worker {process.pid} failed (exit code {process.exitcode}); "
                            "results were not merged."
                        )
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join()
    return merge_results(args.output_dir, num_workers)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="unsloth/Qwen2.5-7B-bnb-4bit")
    parser.add_argument("--data", choices=["mmlu"], default="mmlu")
    parser.add_argument("--output_dir", type=str, default="outputs/mmlu_baseline")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument(
        "--gpu_ids", nargs="+",
        help="One model replica per GPU; defaults to visible GPUs, or 0 1.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.max_model_len <= 512:
        parser.error("--max_model_len must exceed the 512-token output limit")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu_memory_utilization must be in (0, 1]")
    if args.gpu_ids is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        args.gpu_ids = visible.split(",") if visible is not None else ["0", "1"]
    args.gpu_ids = [gpu_id.strip() for gpu_id in args.gpu_ids]
    if any(not gpu_id or gpu_id == "-1" for gpu_id in args.gpu_ids):
        parser.error("--gpu_ids must select at least one visible GPU")
    if len(set(args.gpu_ids)) != len(args.gpu_ids):
        parser.error("--gpu_ids must not contain duplicate devices")

    run_parallel(args)



if __name__ == "__main__":
    main()
