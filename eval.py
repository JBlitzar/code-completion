import torch
#from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
import sys
import time
from dataset import dataset, get_train_dataset

EXPERIMENT_DIRECTORY = "runs/code-decoder-v10-vanilla-smaller-batchfirst"#"runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

device = "mps" if torch.backends.mps.is_available() else "cpu"

device = "cpu"

# net = DecoderTransformer(vocab_size=199, num_blocks=1)
net = make_model()
net.to(device)

net.load_state_dict(
    torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "best.pt"), weights_only=True)
)


for name, param in net.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
for name, param in net.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN found in gradients of {name}")


pad_token_id = 0
sep_token_id = None

input_text = input("Prompt: ")
max_length = 100


input_ids = torch.tensor(dataset.manager.encode(input_text), dtype=int)
print(input_ids.shape)
attention_mask = dataset.manager.attention_mask(input_ids.squeeze(0)).to(device)


#input_ids, attention_mask = get_train_dataset()[4]


# input_ids = input_ids.to(device).unsqueeze(0)

# attention_mask = attention_mask.to(device)  # .unsqueeze(0)

# print(input_ids.shape)

generated_text = dataset.manager.decode(input_ids)

print(generated_text)


for _ in range(max_length):
    with torch.no_grad():
        outputs = net(input_ids)#, padding_mask=attention_mask)

        logits = outputs

        next_token_id = torch.argmax(logits.view(-1, logits.size(-1))[-1], dim=1)


        # next_token_logits = logits.squeeze(1)

        # probs = torch.softmax(next_token_logits, dim=-1)

        # if pad_token_id is not None:
        #     probs[:, pad_token_id] = 0.0

        # if sep_token_id is not None:
        #     probs[:, sep_token_id] = 0.0

        # # repetition_penalty = 2  # Values > 1.0 penalize repetition
        # # for token in input_ids.tolist():
        # #     probs[0, token] /= repetition_penalty

        # # print(probs)

        # next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # Ensure the next_token_id has the same batch size (1 in this case)
        next_token_id = next_token_id.unsqueeze(0)  # Shape: (1, 1)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

        #input_ids = torch.cat((input_ids, next_token_id.unsqueeze(-1)), dim=1)

        attention_mask = torch.cat(
            (attention_mask, torch.ones((1,), device=device)), dim=0
        )

        predicted_token = dataset.manager.decode(next_token_id)
        print(predicted_token)

print()
print()
print(generated_text)
