from datasets import (
    load_dataset,
)  # How presumptuous to have HF call their dataset library "datasets"
import os


dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")


def save_preprocessed(output_file, split="train"):
    data_split = dataset[split]

    separator = " <EOF> "
    all_text = separator.join(data_split["text"])

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)


save_preprocessed(
    os.path.expanduser("~/torch_datasets/wikitext/train/data/corpus_processed.txt")
)
