import math
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from cs336_alignment.sft.data_loader import (
    SFTDataset,
    get_batches
)
from torch.utils.data import DataLoader


def get_model_and_tokenizer(
    model_id: str,
    task_type: str="CAUSAL_LM",
    r: int=16,
    lora_alpha: int=32,
    lora_dropout: float=0.05,
    target_modules: str="all-linear",
    bias: str="none",
):
    # load model
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
    )

    model.config.use_cache = False

    # 1. freeze the base model
    model = prepare_model_for_kbit_training(model)

    # 2. add LoRA config
    config = LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )

    # 3. get lora model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return tokenizer, model

def sft_train(
    model: AutoModelForCausalLM,
    optimizer: torch.optim,
    dataloader: DataLoader,
    gradient_accumulation_steps: int,
    clipping: float,
    train_steps: int
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    sheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(0.03 * train_steps),
        num_training_steps=train_steps
    )

    step = 0
    total_loss = 0
    for index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")

        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1)
        ) / gradient_accumulation_steps

        total_loss += loss.item()
        loss.backward()

        if (index + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
            sheduler.step()

            optimizer.zero_grad()
            step += 1
            print( "step:", step, "loss:", total_loss)
            total_loss = 0
        if step == train_steps:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen2.5-7B-bnb-4bit")
    parser.add_argument("--dataset_path", type=str, default="data/sft/train.jsonl")
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument(
        "--shuffle_document", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--shuffle_dataset", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clipping", type=float, default=1.0)

    args = parser.parse_args()
    tokenizer, model = get_model_and_tokenizer(args.model_id)
    dataset = SFTDataset(tokenizer, args.dataset_path, args.seq_length, args.shuffle_document, args.seed)
    dataloader = get_batches(dataset, args.batch_size // args.gradient_accumulation_steps, args.shuffle_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sft_train(model, optimizer, dataloader, args.gradient_accumulation_steps, args.clipping, args.steps)

if __name__ == "__main__":
    main()
