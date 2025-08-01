import torch

# from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
from dataset import dataset, get_train_dataset, get_dataloader
import torch.nn.functional as F
from tqdm import trange

EXPERIMENT_DIRECTORY = "runs/code-decoder-v23-mega"  # "runs/code-decoder-v22-bigset-tuner"  # "runs/code-decoder-v21-alltrains-tuner"#"runs/code-decoder-v19-bigset-5k"#"runs/code-decoder-v18-allTrains-customTokenizer"#"runs/code-decoder-v17-bpe-upscale"#"runs/code-decoder-v16-upscale"#"runs/code-decoder-v13-rescaling-smaller-retrained"  # "runs/code-decoder-v12-dummy"  # "runs/code-decoder-v11-vanilla-alphabet"#"runs/code-decoder-v10-vanilla-smaller-batchfirst"#"runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

device = "mps" if torch.backends.mps.is_available() else "cpu"

device = "cpu"


def evaluate_topk(model, start_sequence, amt=10, k=20, temperature=1.0, device="cpu"):
    generated_sequence = start_sequence.clone().to(device)

    model.eval()
    with torch.no_grad():
        for _ in trange(amt, leave=False, dynamic_ncols=True, desc="topk"):
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


def evaluate_topp(model, start_sequence, amt=10, p=0.9, temperature=1.0, device="cpu"):
    generated_sequence = start_sequence.clone().to(device)

    model.eval()
    with torch.no_grad():
        for _ in trange(amt, leave=False, dynamic_ncols=True, desc="topp"):
            seq = generated_sequence
            results = model(seq, transpose=True)
            results = results.transpose(0, 1)

            logits = results.reshape(-1, results.size(-1))[-1]
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff_idx = torch.where(cumulative_probs > p)[0][0] + 1
            top_p_probs = sorted_probs[:cutoff_idx]
            top_p_indices = sorted_indices[:cutoff_idx]

            # Normalize selected probabilities
            top_p_probs /= top_p_probs.sum()

            # Sample from the top-p tokens
            sampled_index = torch.multinomial(top_p_probs, 1).item()
            next_token = top_p_indices[sampled_index].unsqueeze(0)

            generated_sequence = torch.cat(
                (generated_sequence, next_token.unsqueeze(0)), dim=1
            )

    return generated_sequence


def evaluate_beam(model, start_sequence, k=2, amt=10, temperature=0.8, device="cpu"):
    generated_sequence = start_sequence.clone().to(device)

    model.eval()

    # Initialize beam candidates (shape: [k, seq_len])
    current_beams = generated_sequence.expand(k, -1)
    current_beam_scores = torch.zeros(k, device=device)

    with torch.no_grad():
        for _ in trange(amt, leave=False, dynamic_ncols=True, desc="beam"):
            all_candidates = []

            # Process each beam
            for i in range(k):
                seq = current_beams[i].unsqueeze(0)  # Shape: [1, seq_len]
                results = model(seq, transpose=True)
                results = results.transpose(0, 1)  # Ensure batch-first shape

                logits = results[:, -1, :] / temperature  # Last token logits
                topk_values, topk_indices = torch.topk(logits, k)  # Shape: [1, k]

                # Expand beam by top-k choices
                for j in range(k):
                    candidate = torch.cat((seq, topk_indices[:, j].unsqueeze(0)), dim=1)
                    score = current_beam_scores[i] + topk_values[:, j]
                    all_candidates.append((candidate, score))

            # Select top-k sequences
            all_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            top_candidates = all_candidates[:k]  # Keep top-k

            current_beams = torch.cat([candidate for candidate, _ in top_candidates])
            current_beam_scores = torch.tensor(
                [score.item() for _, score in top_candidates], device=device
            )

    return current_beams[0]  # Return the best beam sequence


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

    # def tester_exactly_like_trainingmanager_just_next_given_seq_pls(model, seq):
    #     seq = seq.unsqueeze(0)

    #     results = model(batch, transpose=True)
    #     results = results.transpose(0, 1)

    return torch.argmax(results.reshape(-1, results.size(-1)), dim=1)[-1]


def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1)  # Entropy, I guess
    return entropy.mean().item()


def main():
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
    loader = get_dataloader(get_train_dataset())
    torch.random.manual_seed(
        sum([ord(i) for i in input("seed? ")])
    )  # so people can write whatever there
    for data in loader:
        batch, attn_mask = data

        print(
            tester_exactly_like_trainingmanager_please_please_work(net, rawbatch=batch)
        )
        print("pretty please")

        print(
            tester_exactly_like_trainingmanager_only_last_please_work(
                net, rawbatch=batch
            )
        )
        print("please please please")

        # print(
        #     tester_exactly_like_trainingmanager_just_next_given_seq_pls(
        #         net, seq=batch[:, :-1].contiguous()[-1]
        #     )
        # )
        # print(f"Answer was {batch[:,1:].contiguous()[-1][-1]}")
        # print("please please please")

        # print(
        #     tester_exactly_like_trainingmanager_just_next_given_seq_pls(
        #         net, seq=batch[:, :-1].contiguous()[-1][:10]
        #     )
        # )
        # print(f"Answer was {batch[:,1:].contiguous()[-1][10]}")
        # print("please please please")

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
        with torch.no_grad():
            logits = net(batch.unsqueeze(0))  # Pass batch through model
            entropy = compute_entropy(
                logits[:, -1, :]
            )  # Compute entropy at last token position

        print(f"Entropy of last token: {entropy:.4f}")
        # print("USING TOPK")
        # result = evaluate_topk(net, batch.unsqueeze(0), amt=100)
        # print(result)
        # print(
        #     dataset.manager.decode(result[0]),
        #     " | PREFIX FROM TRAIN DSET:",
        #     dataset.manager.decode(batch),
        # )

        print("USING BEAM")
        result = evaluate_beam(net, batch.unsqueeze(0), amt=100, k=3)

        result = dataset.manager.decode(result)
        batch_str = dataset.manager.decode(batch)

        result = f"<data>\n{batch_str}</data>\n{result[len(batch_str) :]}"

        print(result)

        # print(dataset.manager.raw_decode(81))

        break


if __name__ == "__main__":
    main()
