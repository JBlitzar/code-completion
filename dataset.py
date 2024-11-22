import torchvision.datasets as dset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from shutil import copyfile
import subprocess
import youtokentome as yttm


class BPEModelManager:
    def __init__(self, root_dir, vocab_size=5000):
        self.root_dir = root_dir
        self.vocab_size = vocab_size
        self.model_path = os.path.join(root_dir, "bpe_model.model")

        try:
            self.bpe = yttm.BPE(model=self.model_path)
            if self.bpe.vocab_size() != vocab_size:
                print(
                    f"Vocab size mismatch: Expected {vocab_size}, got {self.bpe.vocab_size()}. Retraining model."
                )
                self._backup_model()
                raise ValueError
        except ValueError:
            self._train_bpe_model()

    def _backup_model(self):
        backup_path = os.path.join(self.root_dir, "bpe_model.model.old")
        copyfile(self.model_path, backup_path)

    def _train_bpe_model(self):
        data_path = os.path.join(self.root_dir, "data/corpus.txt")
        processed_path = os.path.join(self.root_dir, "data/corpus_processed.txt")

        with open(data_path, "r") as reader:
            raw_text = reader.read()

        processed_text = self.preprocess_text(raw_text)

        with open(processed_path, "w") as writer:
            writer.write(processed_text)

        yttm.BPE.train(
            data=processed_path, vocab_size=self.vocab_size, model=self.model_path
        )

    def preprocess_text(self, text):
        return text.lower().replace("\t", "    ")

    def encode(self, text: str):
        return self.bpe.encode([text], output_type=yttm.OutputType.ID)

    def decode(self, ids):
        return self.bpe.decode(ids)

    @staticmethod
    def attention_mask(encoded_sequence, mask_token_ids=[0, 1, 2, 3]):
        return [1 if token not in mask_token_ids else 0 for token in encoded_sequence]


class CodeBPEModelManager(BPEModelManager):
    def __init__(self, root_dir, vocab_size=5000):
        super().__init__(root_dir, vocab_size)

    def preprocess_text(self, text):
        formatted_text = self.format_code(text)
        processed_text = formatted_text.replace("\t", "    ").replace("    ", "𝐓")
        return processed_text

    def format_code(self, code):
        try:
            temp_file = os.path.join(self.root_dir, "temp_code.py")
            with open(temp_file, "w") as file:
                file.write(code)

            subprocess.run(["black", temp_file, "--quiet"], check=True)

            with open(temp_file, "r") as file:
                formatted_code = file.read()

            return formatted_code
        except Exception as e:
            print(
                f"Error during code formatting: {e}. Proceeding with unformatted code."
            )
            return code


class TextCorpusDataset(Dataset):
    def __init__(
        self,
        root_dir=os.path.expanduser("~/torch_datasets/github-python/corpus"),
        train=False,
        max_length=512,
        vocab_size=10000,
        IS_CODE=False,
    ):
        self.root = root_dir
        self.manager = BPEModelManager(
            root_dir=root_dir, vocab_size=vocab_size, IS_CODE=False
        )
        self.max_length = max_length

        with open(os.path.join(root_dir, "data/corpus.txt"), "r") as file:
            text = file.read()

            encoded = self.manager.encode(text)[0]
            self.chunks = [
                encoded[i : i + self.max_length]
                for i in range(0, len(encoded), self.max_length)
            ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seq = self.chunks[idx]

        return seq, self.manager.attention_mask(seq)  # todo: convert to tensor.


dataset = TextCorpusDataset(root_dir="./test-data/")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


def get_train_dataset():
    return train_dataset


def get_test_dataset():

    return test_dataset


def get_dataloader(dataset, batch_size=64):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    d = get_train_dataset()
    print("Number of samples: ", len(d))

    a, b = d[4]
    manager = BPEModelManager("./test-data/", vocab_size=10000)
    print(manager.decode(a))
    print()
