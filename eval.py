import torch
from transformers import AutoTokenizer
from architecture import Transformer
import os
import sys
import time

EXPERIMENT_DIRECTORY = "runs/shakespeare-test"  # shakespeare-test, run1-python

device = "mps" if torch.backends.mps.is_available() else "cpu"


net = Transformer()
net.to(device)

net.load_state_dict(
    torch.load(
        os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "latest.pt"), weights_only=True
    )
)


for name, param in net.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
for name, param in net.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN found in gradients of {name}")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


pad_token_id = tokenizer.pad_token_id
sep_token_id = tokenizer.sep_token_id

input_text = input("Prompt: ")
max_length = 100


encoding = tokenizer.encode_plus(
    input_text,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)


generated_text = input_text


for _ in range(max_length):
    with torch.no_grad():
        outputs = net(input_ids, padding_mask=attention_mask)

        logits = outputs

        next_token_logits = logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)

        if pad_token_id is not None:
            probs[:, pad_token_id] = 0.0

        if sep_token_id is not None:
            probs[:, sep_token_id] = 0.0

        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(-1)), dim=1)

        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), device=device)), dim=1
        )

        predicted_token = tokenizer.decode(next_token_id.item())

        if not predicted_token.startswith("##") and not predicted_token.startswith(" "):
            predicted_token += " "
        generated_text += predicted_token.replace("##", "")

        sys.stdout.write(predicted_token.replace("##", ""))
        sys.stdout.flush()

print()
print()
print(generated_text)
