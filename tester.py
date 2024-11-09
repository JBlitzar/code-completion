import torch
from transformers import AutoTokenizer
from architecture import Transformer

# Initialize the Transformer model
t = Transformer()
t.to("mps")  # Move model to MPS (Metal Performance Shaders) for Apple M1/M2/M3 devices

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Sample code sequence
codes = ["def func(a, b):", "if x > 0:", "for i in range(10):"]

# Tokenize and pad sequences
encoding = tokenizer(codes, padding=True, truncation=True, return_tensors="pt")

# This will return a dictionary with 'input_ids', 'attention_mask', etc.
input_ids = encoding['input_ids']  # Tokenized input sequences
attention_mask = encoding['attention_mask']  # Attention mask (1 for real tokens, 0 for padding)

# Print the tokenized input and attention mask
print("Input IDs:")
print(input_ids)
print("Attention Mask:")
print(attention_mask)

# Now, run the transformer model
# Pass the input through the transformer model
# The model should handle the padding mask internally in the attention layers

output = t(input_ids.to("mps"), padding_mask=attention_mask.to("mps"))

# Print the output from the model
print("Transformer output:")
print(output)
