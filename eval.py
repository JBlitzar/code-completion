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

input = torch.randint(199, (1, 1), dtype=torch.long).to(device)

temperature = 1.0

with open("output.txt", 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(100):
            output = net(input, False)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)

            word = dataset.manager.decode(word_idx)

            outf.write(word + ('\n' if i % 20 == 19 else ' '))
