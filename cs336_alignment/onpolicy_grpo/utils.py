import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
from cs336_alignment.checkpoint import get_model_and_tokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor] :
    prompt_output_tokens = []
    response_mask = []
    length = len(prompt_strs)

    for i in range(length):
        prompt = prompt_strs[i]
        output = output_strs[i]
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)

        tokens = prompt_tokens + output_tokens
        prompt_output_tokens.append(tokens)
        response_mask.append([False] * (len(prompt_tokens) - 1) + [True] * len(output_tokens))

    token_sequences = [
        torch.tensor(token_ids, dtype=torch.long)
        for token_ids in prompt_output_tokens
    ]
    mask_sequences = [
        torch.tensor(mask_ids, dtype=torch.bool)
        for mask_ids in response_mask
    ]

    token_ids = pad_sequence(
        token_sequences,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    mask_ids = pad_sequence(
        mask_sequences,
        batch_first=True,
        padding_value=False
    )
    input_ids = token_ids[..., :-1]
    label_ids = token_ids[..., 1:]
    return {
        'input_ids': input_ids,
        'labels': label_ids,
        'response_mask': mask_ids
    }

def get_reponse_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits
    all_log_probs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    batch_size, seq_len = input_ids.shape
    batch_id = torch.arange(0, batch_size, device=logits.device)[:, None]
    seq_id = torch.arange(0, seq_len, device=logits.device)[None, :]
    log_probs = all_log_probs[batch_id, seq_id, labels]

    if return_token_entropy:
        # logits.shape: (batch_size, sequence_length, vocab_size)
        # token_entropy.shape: (batch_size, sequence_length)
        token_entropy = torch.distributions.Categorical(
            logits=logits.float()
        ).entropy()
        return {
            'log_probs': log_probs,
            'token_entropy': token_entropy
        }
    else:
        return {
            'log_probs': log_probs
        }

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "allenai/OLMo-2-0425-1B"
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    a = ["hello", "hello world"]
    b = ["good good", "study, day day up"]

    print(tokenize_prompt_and_output(a, b, tokenizer))

