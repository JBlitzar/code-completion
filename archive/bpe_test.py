import random

import youtokentome as yttm

bpe = yttm.BPE(model="e.model")

train_data_path = "train_data.txt"
model_path = "example.model"

# Generating random file with training data
# 10000 lines with 100 characters in each line
n_lines = 10000
n_characters = 100
with open(train_data_path, "w") as fout:
    for _ in range(n_lines):
        print("".join([random.choice("abcd ") for _ in range(n_characters)]), file=fout)

# Generating random text
test_text = "".join([random.choice("abcde ") for _ in range(100)])

# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)

# Loading model
bpe = yttm.BPE(model=model_path)

# Two types of tokenization
print(bpe.encode([test_text], output_type=yttm.OutputType.ID))
print(bpe.encode([test_text], output_type=yttm.OutputType.SUBWORD))