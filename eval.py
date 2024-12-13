import torch
#from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
import sys
import time
from dataset import dataset, get_train_dataset
import torch.nn.functional as F

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


generated_text = dataset.manager.decode(input_ids)

print(generated_text)
input_ids = torch.randint(199, (1, 1), dtype=torch.long).to(device)  # Initial token
temp = 1.0
generated_text = dataset.manager.decode(input_ids[0].tolist())  # Decode initial token

for _ in range(max_length):
    with torch.no_grad():
        output = net(input_ids)  # Model forward pass
        logits = output[-1, :]  # Get logits for the last token
        word_weights = F.softmax(logits.div(temp), dim=-1).cpu().exp()  # Normalize to probabilities
        word_idx = torch.multinomial(word_weights, 1).item()  # Sample a word index

        word_tensor = torch.tensor([[word_idx]], dtype=torch.long).to(device)  # Wrap sampled token
        input_ids = torch.cat([input_ids, word_tensor], dim=1)  # Append to sequence

        predicted_token = dataset.manager.decode(word_idx)
        print(predicted_token, end="")  # Print as it generates
        generated_text += " " + predicted_token

print("\n\nFINAL TEXT:")
print(generated_text)
