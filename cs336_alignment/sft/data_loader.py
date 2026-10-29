import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_path: str,
        seq_length: int,
        shuffle: bool,
        seed: int = 42
    ):
        with open(
            "cs336_alignment/prompts_safety/alpaca_sft.prompt",
            encoding="utf-8"
        ) as f:
            template = f.read()

        rows = []
        with open(
            dataset_path,
            encoding="utf-8"
        ) as f:
            for row in f:
                rows.append(json.loads(row))

        template_rows = [
            template.format(
                instruction=row["prompt"],
                response=row["response"]
            )
            for row in rows
        ]
        if shuffle:
            random.Random(seed).shuffle(template_rows)

        self.input_ids = []
        self.labels = []

        tokens = []
        for row in template_rows:
            row_token = tokenizer.encode(row.rstrip())
            tokens.extend(row_token)
            tokens.append(tokenizer.eos_token_id)

        for i in range(0, len(tokens), seq_length):
            if i + seq_length >= len(tokens):
                break
            self.input_ids.append(tokens[i: i + seq_length])
            self.labels.append(tokens[i + 1: i + seq_length + 1])

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict[torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[i]),
            "labels": torch.tensor(self.labels[i])
        }


def get_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool
):
    return DataLoader(dataset, batch_size, shuffle)



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    dataset = SFTDataset(tokenizer, "tests/fixtures/sft_sample.jsonl", 100, False)
    dataloader = get_batches(dataset, 2, True)
    for data in dataloader:
        print(data["input_ids"])
        print(data["labels"])