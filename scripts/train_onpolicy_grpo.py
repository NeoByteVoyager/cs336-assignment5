import json
import torch
import random
from transformers import PreTrainedTokenizer, PreTrainedModel

def get_rollout_questions_and_truths(
    train_data: list[dict],
    rollout_batch_size: int,
    group_size: int,
) -> tuple[list[str], list[str]]:
    raw_batch = random.sample(train_data, k=rollout_batch_size // group_size)
    questions = []
    ground_truths = []

    for raw_sample in raw_batch:
        question = raw_sample["question"]
        ground_truth = raw_sample["answer"].split("####")[-1].strip()
        for _ in range(group_size):
            questions.append(question)
            ground_truths.append(ground_truth)

    return questions, ground_truths


def train(
    train_data_path: str,
    test_data_path: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim,
    steps: int,
    rollout_batch_size: int,
    group_size: int,
    generate_batch_size: int,
    seed: int = 42,
):
    random.seed(seed)
    # open train data
    train_data = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    # train step
    for step in range(steps):
        # sample question and generate response
        model.eval()
        questions, ground_truths = get_rollout_questions_and_truths(train_data, rollout_batch_size, group_size)
        responses = []
        assert rollout_batch_size % generate_batch_size == 0
        for i in range(0, rollout_batch_size, generate_batch_size):
            question_microbatch = questions[i: i + generate_batch_size]


if __name__ == "__main__":
    train_data_path = "data/gsm8k/train.jsonl"
    test_path = "data/gsm8k/test.jsonl"
    train_data = []
    with open(train_data_path, "r", encoding="utf-8") as f:
        for line in f:
            train_data.append(json.loads(line))
    q, gt = get_rollout_questions_and_truths(train_data, 4, 2)
    for _ in q:
        print(_)
    for _ in gt:
        print(_)
    pass
