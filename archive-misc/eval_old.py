import torch

# from architecture import DecoderTransformer
from builtin_architecture import make_model
import os
from dataset import dataset
import torch.nn.functional as F

EXPERIMENT_DIRECTORY = "runs/code-decoder-v10-vanilla-smaller-batchfirst"  # "runs/code-decoder-v9-vanilla-smaller"#"runs/code-decoder-v8-smaller"  # "runs/code-decoder-v4-improved"  # shakespeare-test, run1-python

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
generated_text = ""
input_ids = torch.randint(199, (1, 1), dtype=torch.long).to(device)

net.eval()  # Set model to evaluation mode
temp = 1.0  # Balanced temperature

for _ in range(max_length):
    with torch.no_grad():
        output = net(input_ids)  # Model output
        logits = F.log_softmax(output[-1], dim=-1)  # Normalize logits
        word_weights = logits.div(temp).cpu()  # Scale by temperature

        # Top-k sampling
        top_k = 10  # Adjust based on your vocabulary size
        vocab_size = word_weights.size(0)
        top_k = min(top_k, vocab_size)  # Ensure top_k is valid

        top_probs, top_indices = torch.topk(word_weights, k=top_k)

        # Handle edge case: only one valid token
        if top_probs.size(0) == 1:
            word_idx = top_indices[0]  # Directly choose the only available token
        else:
            sampled_idx = torch.multinomial(top_probs, 1).item()
            word_idx = top_indices[sampled_idx]

        # Decode and append token
        print(word_idx)
        predicted_token = dataset.manager.decode(word_idx.item())
        print(predicted_token, end=" ")
        generated_text += predicted_token

        print("Word Weights:", word_weights)
        print("Top Probabilities:", top_probs)
        print("Top Indices:", top_indices)

        # Update input sequence
        word_tensor = torch.tensor([[word_idx]], dtype=torch.long).to(device)
        input_ids = torch.cat([input_ids, word_tensor], dim=1)

print("\nGenerated text:", generated_text)
with open("output.txt", "w+") as f:
    f.write(generated_text)
