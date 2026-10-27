import csv
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cs336_alignment.sft.parse_mmlu_response import parse_mmlu_response


def load_model(
    model_id: str
)-> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()

    return tokenizer, model

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

def get_response(
    tokenizer: AutoTokenizer | None,
    model: AutoModelForCausalLM | None,
    prompt: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = outputs[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)

def evaluate(
    tokenizer:AutoTokenizer | None,
    model: AutoModelForCausalLM | None,
    data: str,
    output_dir: str = "outputs/mmlu_baseline",
):
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
                    for line in csv.reader(f):
                        question = line[0]
                        options = line[1:5]
                        answer = line[5]
                        # get the prompt using template
                        prompt = get_templated_mmlu_prompt(mmlu_zero_shot_template,
                                                           zero_shot_system_template,
                                                           subject,
                                                           question,
                                                           options)
                        response = get_response(
                            tokenizer,
                            model,
                            prompt
                        )
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
        acc = total_correct / total
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



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="unsloth/Qwen2.5-7B-bnb-4bit")
    parser.add_argument("--data", type=str, default="mmlu")
    parser.add_argument("--output_dir", type=str, default="outputs/mmlu_baseline")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_id)
    evaluate(tokenizer, model, args.data, args.output_dir)



if __name__ == "__main__":
    main()
