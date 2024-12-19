import torch

# from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
import sys
import time
from dataset import dataset, get_train_dataset, get_dataloader
import torch.nn.functional as F

EXPERIMENT_DIRECTORY = "runs/code-decoder-v13-rescaling-smaller-retrained"#"runs/code-decoder-v12-dummy"  # "runs/code-decoder-v11-vanilla-alphabet"#"runs/code-decoder-v10-vanilla-smaller-batchfirst"#"runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

device = "mps" if torch.backends.mps.is_available() else "cpu"

device = "cpu"

# net = DecoderTransformer(vocab_size=199, num_blocks=1)
net = make_model()
net.to(device)

net.load_state_dict(
    torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "latest.pt"), weights_only=True)
)


for name, param in net.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
for name, param in net.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN found in gradients of {name}")


def evaluate(model, start_sequence, manager, amt=10, temperature=0.1, window_size=10, argmax=True, k=3):
    #model.eval()
    generated_sequence = start_sequence.clone()
    generated_sequence = generated_sequence.to(device)

    with torch.no_grad():
        for _ in range(amt):
            input_sequence = generated_sequence#[-window_size:] # last window_size amount of tokens


            output = model(input_sequence, transpose=True)
            
            
            # print(f"ARGMAZX: {torch.argmax(output.reshape(-1, output.size(-1)), dim=1)[-1]}")
            # print(probs)
            if argmax:
                output = output.transpose(0,1)
                next_token = torch.argmax(output.reshape(-1, output.size(-1)), dim=1)[-1].unsqueeze(0).unsqueeze(0)
                #print(next_token)
                
            else:
                logits = output[-1, :, :]


                output = output.transpose(0, 1)

                logits = logits / temperature

                # probs = torch.nn.functional.softmax(logits, dim=-1)
                # next_token = torch.multinomial(probs, 1)
                # next_token = next_token.transpose(0, 1)


                topk = torch.topk(logits, k=k)
                values, indeces = topk
                probs = torch.nn.functional.softmax(values, dim=-1)

                next_token = torch.multinomial(probs, 1)
                next_token = next_token.transpose(0, 1)


            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)
    final = manager.decode(generated_sequence.squeeze(0))
    return final


def tester_exactly_like_trainingmanager_please_please_work(model, rawbatch):
    labels = rawbatch[:, 1:].contiguous()
    batch = rawbatch[:, :-1].contiguous()
    results = model(batch, transpose=True)
    results = results.transpose(0, 1)
    print(torch.sum(
            torch.argmax(results.reshape(-1, results.size(-1)), dim=1)
            == labels.reshape(-1)
        ) / len(labels.reshape(-1)))
    return torch.argmax(results.reshape(-1, results.size(-1)), dim=1), labels.reshape(-1)
loader = get_dataloader(get_train_dataset())
for data in loader:
    batch, attn_mask = data

    print(tester_exactly_like_trainingmanager_please_please_work(net, rawbatch=batch))
    print("pretty please")

    labels = batch[:, 1:].contiguous()
    batch = batch[:, :-1].contiguous()

    batch = batch[0]
    labels = labels[0]
    
    # inp, mask = dataset[0]

    # inp = inp[:-1]
    print(batch)
    print(dataset.manager.decode(batch))
    print(dataset.manager.decode(labels))
    print("that's inp I guess ^^")

    print(
        evaluate(net, batch.unsqueeze(0), dataset.manager, argmax=False)
    )
    exit()
