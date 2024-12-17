import torch

# from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
import sys
import time
from dataset import dataset, get_train_dataset
import torch.nn.functional as F

EXPERIMENT_DIRECTORY = "runs/code-decoder-v12-dummy"  # "runs/code-decoder-v11-vanilla-alphabet"#"runs/code-decoder-v10-vanilla-smaller-batchfirst"#"runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

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


def evaluate(model, start_sequence, amt=1, temperature=0.1, window_size=10):
    model.eval()
    generated_sequence = start_sequence.clone()
    batch_size = start_sequence.size(1)
    device = next(model.parameters()).device
    generated_sequence = generated_sequence.to(device)

    with torch.no_grad():
        for _ in range(amt):
            input_sequence = generated_sequence[-window_size:]
            output = model(input_sequence, transpose=True)
            logits = output[-1, :, :]
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output = output.transpose(0, 1)
            # print(f"ARGMAZX: {torch.argmax(output.reshape(-1, output.size(-1)), dim=1)[-1]}")
            # print(probs)
            next_token = torch.multinomial(probs, 1)
            next_token = next_token.transpose(0, 1)
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

    return generated_sequence


print(
    evaluate(net, torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32).unsqueeze(0))
)
exit()
