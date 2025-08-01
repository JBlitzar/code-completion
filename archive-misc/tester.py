from transformers import AutoTokenizer
from architecture import Transformer


t = Transformer()
t.to("mps")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


codes = ["def func(a, b):", "if x > 0:", "for i in range(10):"]


encoding = tokenizer(codes, padding=True, truncation=True, return_tensors="pt")


input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]


print("Input IDs:")
print(input_ids)
print("Attention Mask:")
print(attention_mask)

output = t(input_ids.to("mps"), padding_mask=attention_mask.to("mps"))


print("Transformer output:")
print(output)
