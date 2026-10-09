import json
import argparse
from pathlib import Path
from copy import deepcopy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM



from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
def load():
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", device_map="auto")
    pipe = pipeline("text-generation", model="allenai/OLMo-2-0425-1B")

    config = deepcopy(model.generation_config)
    config.max_length = None
    config.max_new_tokens = 512
    config.do_sample = True
    config.temperature = 1.0
    config.top_p = 1.0
    config.top_k = 0  # 关闭额外的 top-k 筛选
    config.stop_strings = ["</answer>"]
    return pipe, config, model, tokenizer

def inference(pipe, config, tokenizer, prompt_path, data_path, is_r1):
    template = Path(
        prompt_path
    ).read_text()
    with open(data_path, "r") as f:
        i = 0
        for line in f:
            data = json.loads(line)
            question = data["question"]
            prompt = template.format(question=question)
            print(prompt)
            result = pipe(
                prompt,
                generation_config=config,
                tokenizer = tokenizer,
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )

            answer = result[0]["generated_text"]
            print("ans:", answer)
            ground_truth = data["answer"].split("####")[-1].strip()
            print("ground_truth:", ground_truth)

            if is_r1:
                reward = r1_zero_reward_fn(answer, ground_truth)
            else:
                reward = question_only_reward_fn(answer, ground_truth)
            print("format_reward: ", reward["format_reward"])
            print("answer_reward: ", reward["answer_reward"])
            print("reward: ", reward["reward"])
            print("-" * 80)
            i += 1
            if i >= 10:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str)
    args = parser.parse_args()

    pipe, config, model, tokenizer = load()
    '''
    prompt_path1 = "cs336_alignment/prompts/question_only.prompt"
    prompt_path2 = "cs336_alignment/prompts/r1_zero.prompt"
    prompt_path3 = "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
    '''
    data_path = "data/gsm8k/test.jsonl"
    if args.prompt_path == "cs336_alignment/prompts/question_only.prompt":
        is_r1 = False
    else:
        is_r1 = True
    inference(pipe, config, tokenizer, args.prompt_path, data_path, is_r1)

