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
import re
import time


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
            self.bpe = yttm.BPE(model=self.model_path)

    def _backup_model(self):
        backup_path = os.path.join(self.root_dir, "bpe_model.model.old")
        copyfile(self.model_path, backup_path)

    def _train_bpe_model(self):
        data_path = os.path.join(self.root_dir, "data/corpus.txt")
        processed_path = os.path.join(self.root_dir, "data/corpus_processed.txt")

        with open(data_path, "r", errors="ignore") as reader:
            raw_text = reader.read()

        processed_text = self.preprocess_text(raw_text)

        with open(processed_path, "w") as writer:
            writer.write(processed_text)

        yttm.BPE.train(
            data=processed_path, vocab_size=self.vocab_size, model=self.model_path
        )

    def preprocess_text(self, text):
        return text.lower()

    def encode(self, text: str):
        return self.bpe.encode([text], output_type=yttm.OutputType.ID)

    def decode(self, ids):
        return self.bpe.decode(ids)

    @staticmethod
    def attention_mask(encoded_sequence, mask_token_ids=[0, 1, 2, 3]):
        return [1 if token not in mask_token_ids else 0 for token in encoded_sequence]


class CodeBPEModelManager(BPEModelManager):
    mapping_dict = {
        "    " : " <INDENT> ",
        "\n" : " <NEWLINE> ",
            
        }
    def __init__(self, root_dir, vocab_size=5000):
        super().__init__(root_dir, vocab_size) 


        

    def preprocess_text(self, text):
        print("Formatting....")
        processed_text = self.format_code(text)

        for key, value in CodeBPEModelManager.mapping_dict.items():
            processed_text = processed_text.replace(key, value)

        return processed_text
    
    def encode(self, text: str):
        processed_text = text
        for key, value in CodeBPEModelManager.mapping_dict.items():
            processed_text = processed_text.replace(key, value)

        return self.bpe.encode([processed_text], output_type=yttm.OutputType.ID)

    def decode(self, ids):
        result = self.bpe.decode(ids)[0]
        print(result)
        for key, value in CodeBPEModelManager.mapping_dict.items():
            result = result.replace(value, key) # value, key

       
                
        return result
    
    def _train_bpe_model(self):
        print("Training (1)....")
        data_path = os.path.join(self.root_dir, "data/corpus.txt")
        processed_path = os.path.join(self.root_dir, "data/corpus_processed.txt")

        with open(data_path, "r", errors="ignore", encoding="utf-8") as reader:
            raw_text = reader.read()

        processed_text = self.preprocess_text(raw_text)

        with open(processed_path, "w", encoding="utf-8") as writer:
            writer.write(processed_text)


        print("removing temp file...")
        temp_file = os.path.join(self.root_dir, "temp_code.py") # dont ask
        os.remove(temp_file)


        print("Training....")
        yttm.BPE.train(
            data=processed_path, vocab_size=self.vocab_size, model=self.model_path, coverage=0.999
        )


        

    def format_code(self, code):
        try:
            temp_file = os.path.join(self.root_dir, "temp_code.py")
            with open(temp_file, "w") as file:
                file.write(code.replace("\t", "    ")) # Hacky replacement, black freaks out otherwise

            #subprocess.run(["black", temp_file, "--quiet"], check=True)
            subprocess.run(["autopep8", "--in-place", "--ignore=E402", temp_file], check=True)

            with open(temp_file, "r") as file:
                formatted_code = file.read()
            
            

            return formatted_code
        except Exception as e:
            print(
                f"Error during code formatting: {e}."
            )
            return code


class TextCorpusDataset(Dataset):
    def __init__(
        self,
        root_dir="./test-data",
        train=False,
        max_length=512,
        vocab_size=10000,
        IS_CODE=False,
    ):
        self.root = root_dir
        if IS_CODE:
            self.manager = CodeBPEModelManager(
            root_dir=root_dir, vocab_size=vocab_size
            )
        else:
            self.manager = BPEModelManager(
                root_dir=root_dir, vocab_size=vocab_size
            )
        self.max_length = max_length

        self.cache_file = os.path.join(root_dir, "encoded_cached.pt")

        start_t = time.time()
        if os.path.exists(self.cache_file):
            print("[Dataset] Loading cached chunks...")
            
            encoded = torch.load(self.cache_file, weights_only=True)  # Load cached chunks
            self.chunks = [
                encoded[i : i + self.max_length]
                for i in range(0, len(encoded), self.max_length)
            ]
        else:
            with open(os.path.join(root_dir, "data/corpus_processed.txt"), "r", errors="ignore") as file:
                text = file.read()
                print("encoding...")
                encoded = self.manager.encode(text)[0]
                self.chunks = [
                    encoded[i : i + self.max_length]
                    for i in range(0, len(encoded), self.max_length)
                ]
                
                torch.save(encoded, self.cache_file)
        
        end_t = time.time()
        print(f"Dataset loading took {end_t - start_t} seconds.")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seq = self.chunks[idx]

        return seq, self.manager.attention_mask(seq)  # todo: convert to tensor.

#print("Running....")
dataset = TextCorpusDataset(root_dir=os.path.expanduser("~/torch_datasets/github-python/corpus"), vocab_size=10000, IS_CODE=True)
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
    manager = dataset.manager
    print(manager.decode(a))
    print()
