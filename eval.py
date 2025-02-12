import torch

# from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
import sys
import time
from dataset import dataset, get_train_dataset, get_dataloader
import torch.nn.functional as F
from tqdm import tqdm, trange

EXPERIMENT_DIRECTORY = "runs/code-decoder-v22-bigset-tuner"  # "runs/code-decoder-v21-alltrains-tuner"#"runs/code-decoder-v19-bigset-5k"#"runs/code-decoder-v18-allTrains-customTokenizer"#"runs/code-decoder-v17-bpe-upscale"#"runs/code-decoder-v16-upscale"#"runs/code-decoder-v13-rescaling-smaller-retrained"  # "runs/code-decoder-v12-dummy"  # "runs/code-decoder-v11-vanilla-alphabet"#"runs/code-decoder-v10-vanilla-smaller-batchfirst"#"runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

device = "mps" if torch.backends.mps.is_available() else "cpu"

device = "cpu"

# net = DecoderTransformer(vocab_size=199, num_blocks=1)
net = make_model()
net.to(device)
print(os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "latest.pt"))
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


def evaluate_topk(model, start_sequence, amt=10, k=10, temperature=0.8):
    generated_sequence = start_sequence.clone().to(device)

    model.eval()
    with torch.no_grad():
        for _ in trange(amt, leave=False):
            seq = generated_sequence
            results = model(seq, transpose=True)
            results = results.transpose(0, 1)

            logits = results.reshape(-1, results.size(-1))[-1]

            logits = logits / temperature

            top_k_values, top_k_indices = torch.topk(logits, k)
            top_k_probs = F.softmax(top_k_values, dim=-1)

            sampled_index = torch.multinomial(top_k_probs, 1).item()
            next_token = top_k_indices[sampled_index].unsqueeze(0)

            generated_sequence = torch.cat(
                (generated_sequence, next_token.unsqueeze(0)), dim=1
            )

    return generated_sequence

def evaluate_beam(model, start_sequence, k=2, amt=10):
    generated_sequence = start_sequence.clone().to(device)

    model.eval()

    current_beams = [generated_sequence]
    current_beam_scores = [1.0]

    with torch.no_grad():
        for _ in trange(amt, leave=False, dynamic_ncols=True):

            unpruned_new_beams = []
            unpruned_new_beam_scores = []
            for idx, beam in enumerate(current_beams):
                # generate the top k next tokens for each beam
                # add them to a temp list
                seq = beam
                results = model(seq, transpose=True)
                results = results.transpose(0, 1)

                logits = results.reshape(-1, results.size(-1))[-1]
                # values are probs
                # indices are actual tokens

                # top_k_values, top_k_indices = torch.topk(logits, k)
                topk = torch.topk(logits, k)

                for topk_idx in range(len(topk[0])):
                    value = topk[0][topk_idx]
                    index = topk[1][topk_idx]

                    unpruned_new_beam_scores.append(value + current_beam_scores[idx])

                    unpruned_new_beams.append(torch.cat((beam,index.unsqueeze(0).unsqueeze(0)),dim=1))
                
            beams_and_scores = dict(zip(unpruned_new_beams,unpruned_new_beam_scores))

            top_beams_and_scores = sorted(beams_and_scores.items(),key=lambda x: x[1],reverse=True)[:k]

            current_beams = [a[0] for a in top_beams_and_scores]
            current_beam_scores = [a[1] for a in top_beams_and_scores]

            print(len(current_beams))
            print(current_beams[0].size())
    
    # final k beams

    beams_and_scores = dict(zip(current_beams,current_beam_scores))

    generated_sequence, _ = sorted(beams_and_scores.items(),key=lambda x: x[1],reverse=True)[0]


    return generated_sequence


def evaluate(
    model,
    start_sequence,
    amt=10,
):
    generated_sequence = start_sequence.clone()
    generated_sequence = generated_sequence.to(device)

    model.eval()
    with torch.no_grad():
        for _ in trange(amt, leave=False):
            seq = generated_sequence
            results = model(seq, transpose=True)
            results = results.transpose(0, 1)

            next_token = torch.argmax(results.reshape(-1, results.size(-1)), dim=1)[
                -1
            ].unsqueeze(0)

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
torch.random.manual_seed(
    sum([ord(i) for i in input("seed? ")])
)  # so people can write whatever there
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

    batch = batch[:100]
    labels = labels[:100]
    print("Getting first 100 tokens for batch and labels")

    # inp, mask = dataset[0]

    # inp = inp[:-1]
    print(batch)
    print(dataset.manager.decode(batch))
    print("batch ^ labels v")
    print(dataset.manager.decode(labels))
    print("that's inp I guess ^^")
    # print("USING TOPK")
    # result = evaluate_topk(net, batch.unsqueeze(0), amt=100)
    print("usinb beam")
    result = evaluate_beam(net, batch.unsqueeze(0),amt=100)
    print(result)
    print(
        dataset.manager.decode(result[0]),
        " | PREFIX FROM TRAIN DSET:",
        dataset.manager.decode(batch),
    )

    # print(dataset.manager.raw_decode(81))

    break
