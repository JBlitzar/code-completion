import torch
from transformers import AutoTokenizer
from architecture import Transformer
import os
import sys
import time

EXPERIMENT_DIRECTORY = "runs/run1-python"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Initialize model
net = Transformer()
net.to(device)

# Load checkpoint
net.load_state_dict(torch.load(os.path.join(EXPERIMENT_DIRECTORY, "ckpt", "latest.pt"), weights_only=True))

# Check for NaN values in model parameters and gradients
for name, param in net.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
for name, param in net.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN found in gradients of {name}")

# Use GPT2 tokenizer (or another autoregressive tokenizer)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # or use the appropriate tokenizer for your model

# Get PAD token ID (if it exists) to avoid generating padding tokens
pad_token_id = tokenizer.pad_token_id
sep_token_id = tokenizer.sep_token_id

input_text = input("Prompt: ")
max_length = 100

# Tokenize input text
encoding = tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

input_ids = encoding['input_ids'].to(device)  # Move to the appropriate device
attention_mask = encoding['attention_mask'].to(device)  # Ensure attention mask is also on the device

# Initialize generated text with the input
generated_text = input_text

# Autoregressive generation
for _ in range(max_length):
    with torch.no_grad():
        # Forward pass through the model
        outputs = net(input_ids, padding_mask=attention_mask)  # Ensure padding mask is passed if needed
        
        # Extract the logits for the last token
        logits = outputs  # Assuming `outputs` is the raw logits
        
        next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
        
        probs = torch.softmax(next_token_logits, dim=-1)  # Apply softmax over the last dimension
        
        # Mask out PAD and SEP tokens by setting their probability to 0
        if pad_token_id is not None:
            probs[:, pad_token_id] = 0.0
        
        if sep_token_id is not None:
            probs[:, sep_token_id] = 0.0

        # Sample from the probability distribution instead of using argmax
        next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Append the predicted token to the input sequence
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(-1)), dim=1)

        # Update the attention mask: append 1 for the new token to indicate it's a real token
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), device=device)), dim=1)

        # Decode the token ID to text
        predicted_token = tokenizer.decode(next_token_id.item())

        # Properly handle spaces when adding tokens
        if not predicted_token.startswith("##") and not predicted_token.startswith(" "):
            predicted_token += " "
        generated_text += predicted_token.replace("##", "")
        
        # Print the token as it's generated, simulating typing
        sys.stdout.write(predicted_token.replace("##", ""))
        sys.stdout.flush()

print()
print()
print(generated_text)