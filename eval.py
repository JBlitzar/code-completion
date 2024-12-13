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


generated_text = dataset.manager.decode(input_ids)

print(generated_text)
input_ids = torch.randint(199, (1, 1), dtype=torch.long).to(device)

temp = 0.1

for _ in range(max_length):
    with torch.no_grad():


        output = net(input_ids)
        word_weights = output[-1].squeeze().div(0.1).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        word_tensor = torch.Tensor([[word_idx]]).long().to(device)
        input_ids = torch.cat([input_ids, word_tensor], 0)

        predicted_token = dataset.manager.decode(word_idx)
        print(predicted_token, end="")

        generated_text += predicted_token
        

print()
print()
print("FINAL THING BELOW vvv")
print(generated_text)
