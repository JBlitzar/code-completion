import torch

# from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
import sys
import time
from dataset import dataset, get_train_dataset, get_dataloader
import torch.nn.functional as F

EXPERIMENT_DIRECTORY = "runs/code-decoder-v16-upscale"#"runs/code-decoder-v13-rescaling-smaller-retrained"  # "runs/code-decoder-v12-dummy"  # "runs/code-decoder-v11-vanilla-alphabet"#"runs/code-decoder-v10-vanilla-smaller-batchfirst"#"runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

device = "mps" if torch.backends.mps.is_available() else "cpu"

device = "cpu"

# net = DecoderTransformer(vocab_size=199, num_blocks=1)
net = make_model()
net.to(device)
print( os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "latest.pt"))
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


def evaluate(
    model,
    start_sequence,
    amt=10,
):
    generated_sequence = start_sequence.clone()
    generated_sequence = generated_sequence.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(amt):
            seq = generated_sequence
            results = model(seq, transpose=True)
            results = results.transpose(0, 1)

            next_token = torch.argmax(results.reshape(-1, results.size(-1)), dim=1)[-1].unsqueeze(0)

            generated_sequence = torch.cat(
                (generated_sequence, next_token.unsqueeze(0)), dim=1
            )

    return generated_sequence


def tester_exactly_like_trainingmanager_please_please_work(model, rawbatch):
    labels = rawbatch[:, 1:].contiguous()
    batch = rawbatch[:, :-1].contiguous()
    results = model(batch, transpose=True)
    results = results.transpose(0, 1)
    print(
        torch.sum(
            torch.argmax(results.reshape(-1, results.size(-1)), dim=1)
            == labels.reshape(-1)
        )
        / len(labels.reshape(-1))
    )
    return torch.argmax(results.reshape(-1, results.size(-1)), dim=1), labels.reshape(
        -1
    )


def tester_exactly_like_trainingmanager_only_last_please_work(model, rawbatch):
    labels = rawbatch[:, 1:].contiguous()
    batch = rawbatch[:, :-1].contiguous()

    batch = batch[-1].unsqueeze(0)
    labels = labels[-1].unsqueeze(0)  # works bc my data is initially batch-first

    results = model(batch, transpose=True)
    results = results.transpose(0, 1)
    print(
        torch.sum(
            torch.argmax(results.reshape(-1, results.size(-1)), dim=1)
            == labels.reshape(-1)
        )
        / len(labels.reshape(-1))
    )
    return torch.argmax(results.reshape(-1, results.size(-1)), dim=1), labels.reshape(
        -1
    )


def tester_exactly_like_trainingmanager_just_next_given_seq_pls(model, seq):
    seq = seq.unsqueeze(0)

    results = model(batch, transpose=True)
    results = results.transpose(0, 1)

    return torch.argmax(results.reshape(-1, results.size(-1)), dim=1)[-1]


loader = get_dataloader(get_train_dataset())
for data in loader:
    batch, attn_mask = data

    print(tester_exactly_like_trainingmanager_please_please_work(net, rawbatch=batch))
    print("pretty please")

    print(
        tester_exactly_like_trainingmanager_only_last_please_work(net, rawbatch=batch)
    )
    print("please please please")

    print(
        tester_exactly_like_trainingmanager_just_next_given_seq_pls(
            net, seq=batch[:, :-1].contiguous()[-1]
        )
    )
    print(f"Answer was {batch[:,1:].contiguous()[-1][-1]}")
    print("please please please")

    print(
        tester_exactly_like_trainingmanager_just_next_given_seq_pls(
            net, seq=batch[:, :-1].contiguous()[-1][:10]
        )
    )
    print(f"Answer was {batch[:,1:].contiguous()[-1][10]}")
    print("please please please")

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
    result = evaluate(net, batch.unsqueeze(0))
    print(result)
    print(dataset.manager.decode(result[0]))
    exit()
