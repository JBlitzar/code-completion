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
from tqdm import trange


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
            data=processed_path,
            vocab_size=self.vocab_size,
            model=self.model_path,
            coverage=0.9999,
        )

    def preprocess_text(self, text):
        return text.lower()

    def encode(self, text: str):
        return self.bpe.encode([text], output_type=yttm.OutputType.ID)

    def decode(self, ids):
        return self.bpe.decode(ids.tolist())[0]

    @staticmethod
    def attention_mask(encoded_sequence, mask_token_ids=[0, 1, 2, 3]):
        mask_token_tensor = torch.tensor(mask_token_ids, dtype=torch.int)
        # print(mask_token_tensor)
        # print(encoded_sequence)
        return (encoded_sequence.unsqueeze(1) != mask_token_tensor).all(dim=1).int()


class CodeBPEModelManager(BPEModelManager):
    mapping_dict = {
        "    ": " <INDENT> ",
        "\n": " <NEWLINE> ",
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
        result = self.bpe.decode(ids.tolist())[0]
        # print(result)
        for key, value in CodeBPEModelManager.mapping_dict.items():
            result = result.replace(value.strip(), key)  # value, key

        return result

    def _train_bpe_model(self):
        print("Training (1)....")
        data_path = os.path.join(self.root_dir, "data/corpus.txt")
        processed_path = os.path.join(self.root_dir, "data/corpus_processed.txt")

        if input("Reformat? Will take time [y/N]") == "y":

            with open(data_path, "r", errors="ignore", encoding="utf-8") as reader:
                raw_text = reader.read()

            processed_text = self.preprocess_text(raw_text)

            with open(processed_path, "w", encoding="utf-8") as writer:
                writer.write(processed_text)

            print("removing temp file...")
            temp_file = os.path.join(self.root_dir, "temp_code.py")  # dont ask
            os.remove(temp_file)

        print("Training....")
        yttm.BPE.train(
            data=processed_path,
            vocab_size=self.vocab_size,
            model=self.model_path,
            coverage=0.995,
        )

    def format_code(self, code):
        try:
            temp_file = os.path.join(self.root_dir, "temp_code.py")
            with open(temp_file, "w") as file:
                file.write(
                    code.replace("\t", "    ")
                )  # Hacky replacement, black freaks out otherwise

            subprocess.run(["black", temp_file, "--quiet"], check=True)
            subprocess.run(
                ["autopep8", "--in-place", "--ignore=E402", temp_file], check=True
            )

            with open(temp_file, "r") as file:
                formatted_code = file.read()

            return formatted_code
        except Exception as e:
            print(f"Error during code formatting: {e}.")
            return code
        

class CodeCustomTokenizerManager(BPEModelManager):
    reserved_keywords = [
        "false", "await", "else", "import", "pass", "none", "break", "except",
        "in", "raise", "true", "class", "finally", "is", "return", "and", "continue",
        "for", "lambda", "try", "as", "def", "from", "nonlocal", "while", "assert",
        "del", "global", "not", "with", "async", "elif", "if", "or", "yield"
    ]
    symbols = ["(", ")", "[", "]", "{", "}", ".", ",", ":", ";", "+", "-", "*", "/", "%", "=", "<", ">", "&", "|", "^", "~", "!", "==", "!=", "<=", ">=", "**", "//", "@", "#", "\\", "'", "\""]

    def __init__(self, root_dir, vocab_size=5000):
        self.root_dir = root_dir
        self.token_to_id = {}
        print("This is CodeCustomTokenizerManager, vocab size will be disregarded.")

    def preprocess_text(self, code):
        print("Preprocessing text...")

        code = code.lower().replace("	", "    ")
        
        # comments
        code = re.sub(r"#.*", "", code)
        code = re.sub(r'"""(.*?)"""', "", code, flags=re.DOTALL) # funny usage of re
        code = re.sub(r"'''(.*?)'''", "", code, flags=re.DOTALL)

        # filter non-ascii
        code = re.sub(r"[^\x00-\x7F]+", "", code.lower())

        # each reserved word/symbol is a token. We split by space at the end, so this works.
        for word in self.reserved_keywords:
            code = re.sub(rf"\b{word}\b", f" {word} ", code)
        for symbol in self.symbols:
            code = code.replace(symbol, f" {symbol} ")

        # Split identifiers by spaces, underscores, or capitalization
        def split_token(token):
            result = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
            return result.split("_")

        tokens = []
        for token in re.split(r"(\W)", code):
            if token.strip():
                tokens.extend(split_token(token))

        return [tok for tok in tokens if tok.strip()]
    
    

    def encode(self, code):
        tokens = self.preprocess_text(code)

        for token in tokens:
            if token not in self.token_to_id:
                new_id = len(self.token_to_id)
                self.token_to_id[token] = new_id
                self.id_to_token[new_id] = token

        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids):
        result = self.bpe.decode(ids.tolist())[0]
        # print(result)
        for key, value in CodeBPEModelManager.mapping_dict.items():
            result = result.replace(value.strip(), key)  # value, key

        return result

    def _train_bpe_model(self):
        print("Training (1)....")
        data_path = os.path.join(self.root_dir, "data/corpus.txt")
        processed_path = os.path.join(self.root_dir, "data/corpus_processed.txt")

        if input("Reformat? Will take time [y/N]") == "y":

            with open(data_path, "r", errors="ignore", encoding="utf-8") as reader:
                raw_text = reader.read()

            processed_text = self.preprocess_text(raw_text)

            with open(processed_path, "w", encoding="utf-8") as writer:
                writer.write(processed_text)

            print("removing temp file...")
            temp_file = os.path.join(self.root_dir, "temp_code.py")  # dont ask
            os.remove(temp_file)




    def format_code(self, code):
        try:
            temp_file = os.path.join(self.root_dir, "temp_code.py")
            with open(temp_file, "w") as file:
                file.write(
                    code.replace("\t", "    ")
                )  # Hacky replacement, black freaks out otherwise

            subprocess.run(["black", temp_file, "--quiet"], check=True)
            subprocess.run(
                ["autopep8", "--in-place", "--ignore=E402", temp_file], check=True
            )

            with open(temp_file, "r") as file:
                formatted_code = file.read()

            return formatted_code
        except Exception as e:
            print(f"Error during code formatting: {e}.")
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
        print(root_dir)
        self.root = root_dir
        if IS_CODE:
            self.manager = CodeBPEModelManager(root_dir=root_dir, vocab_size=vocab_size)
        else:
            self.manager = BPEModelManager(root_dir=root_dir, vocab_size=vocab_size)
        self.max_length = max_length

        self.cache_file = os.path.join(root_dir, "encoded_chunked.pt")

        start_t = time.time()
        if os.path.exists(self.cache_file):
            self.chunks = torch.load(self.cache_file, weights_only=True)
            if self.chunks.size(-1) != self.max_length:
                if (
                    input(
                        "Attempting to fix and re-chunk data to correct length. Continue? [y/N]: "
                    )
                    == "y"
                ):
                    self._chunk_and_save(torch.flatten(self.chunks).tolist())
                    print("Re-chunked successfully!")
                else:
                    print("Operation aborted.")
        else:
            with open(
                os.path.join(root_dir, "data/corpus_processed.txt"),
                "r",
                errors="ignore",
            ) as file:
                text = file.read()
                encoded = self.manager.encode(text)[0]

                self._chunk_and_save(encoded)

        end_t = time.time()
        print(f"Dataset loading took {end_t - start_t} seconds.")

    def _chunk_and_save(self, encoded):
        chunked_data = [
            torch.tensor(encoded[i : i + self.max_length], dtype=torch.int)
            for i in trange(0, len(encoded), self.max_length, leave=False)
        ]

        # me when the last item is not necesarily of length self.max_length
        padded_chunk = torch.zeros(self.max_length, dtype=torch.int)
        padded_chunk[: len(chunked_data[-1])] = chunked_data[
            -1
        ]  # silly zero fill bc im *optimized* like that
        chunked_data[-1] = padded_chunk

        self.chunks = torch.stack(chunked_data)
        torch.save(self.chunks, self.cache_file)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seq = self.chunks[idx]
        return seq, self.manager.attention_mask(seq)


# print("Running....")
dataset = TextCorpusDataset(
    root_dir=os.path.expanduser(
        "~/torch_datasets/github-python/all_trains_subset_corpus"
        #"~/torch_datasets/github-python/corpus"
    ),  # os.path.expanduser("~/torch_datasets/wikitext/train")
    vocab_size=500,
    IS_CODE=True,  # Remember to change!
    max_length=50,
)
dset_size = int(len(dataset))
train_size = int(0.8 * dset_size)
test_size = int(dset_size - train_size)

train_dataset, test_dataset, _ = random_split(
    dataset, [train_size, test_size, len(dataset) - train_size - test_size]
)


def get_train_dataset():
    return train_dataset


def get_test_dataset():

    return test_dataset


def get_dataloader(dataset, batch_size=128):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    d = get_train_dataset()
    print("Number of samples: ", len(d))

    a, b = d[4]
    manager = dataset.manager
    print(manager.decode(a))
    print()
