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
import numpy as np


# Device for dataloading and dataloading only. Dataloading on MPS was slower

DEVICE = "cpu"#"mps" if torch.backends.mps.is_available() else "cpu"

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
        mask_token_tensor = torch.tensor(mask_token_ids, dtype=torch.int).to(encoded_sequence.device)
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

        return self.bpe.encode([processed_text], output_type=yttm.OutputType.ID)[0]

    def decode(self, ids):
        # print(ids)
        # print("ids^^")
        l = ids
        if isinstance(l, torch.Tensor):
            l = ids.tolist()
        if isinstance(l, int):
            l = [l]

        result = self.bpe.decode(l)[0]
        # print(result)
        for key, value in CodeBPEModelManager.mapping_dict.items():
            result = result.replace(value.strip(), key)  # value, key

        return result
    
    def raw_decode(self, id: int):
        return self.bpe.decode([id])[0]

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
            coverage=1,
            #coverage=0.995, # TODO: revert if you want
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
        "false",
        "await",
        "else",
        "import",
        "pass",
        "none",
        "break",
        "except",
        "in",
        "raise",
        "true",
        "class",
        "finally",
        "is",
        "return",
        "and",
        "continue",
        "for",
        "lambda",
        "try",
        "as",
        "def",
        "from",
        "nonlocal",
        "while",
        "assert",
        "del",
        "global",
        "not",
        "with",
        "async",
        "elif",
        "if",
        "or",
        "yield",
    ]
    symbols = [
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ".",
        ",",
        ":",
        ";",
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "<",
        ">",
        "&",
        "|",
        "^",
        "~",
        "!",
        "==",
        "!=",
        "<=",
        ">=",
        "**",
        "//",
        "@",
        "#",
        "\\",
        "'",
        '"',
        "`",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0x",
        "0d",
        "0o",

    ]

    def __init__(self, root_dir, vocab_size=5000, cutoff_thresh=0.1, use_vocab_size_instead=False): # keep 90% with thresh 0.1
        self.root_dir = root_dir

        self.token_to_id = {"<PAD>": 0}
        print("This is CodeCustomTokenizerManager, vocab size will be disregarded.")

        print(f"Cutoff threshold: {cutoff_thresh}")
        self.cutoff_thresh = cutoff_thresh

        if use_vocab_size_instead:
            print("Nevermind! Using vocab size instead, no cutoff thresh")

        self.use_vocab_size_instead = use_vocab_size_instead

        self.vocab_size = vocab_size

        vocab_path = os.path.join(self.root_dir, "custom_tokens_vocab.txt")
        try:
            self.load_vocab(vocab_path)
        except FileNotFoundError:
            print("Making vocab!")
            self.make_vocab()
            self.save_vocab(vocab_path)

        print(f"Vocab size: {len(self.token_to_id)}")

    def make_vocab(self):
        data_path = os.path.join(self.root_dir, "data/corpus.txt")
        processed_path = os.path.join(self.root_dir, "data/corpus_processed.txt")

        with open(data_path, "r", errors="ignore") as reader:
            raw_text = reader.read()

        processed_text = self.preprocess_text(raw_text)

        with open(processed_path, "w") as writer:
            writer.write(" ".join(processed_text))

        for token in processed_text:
            if token not in self.token_to_id:
                if len(self.token_to_id) == 0:
                    self.token_to_id = {"<PAD>": 0}  # TODO: bad practice or something

                self.token_to_id[token] = len(self.token_to_id)

        print(f"Number of tokens: {len(self.token_to_id)}")

    def preprocess_text(self, code):
        print("Preprocessing text...", code[:20])

        
        # print(code[:100])

        # comments
        code = re.sub(r"#.*", "", code)
        code = re.sub(r'"""(.*?)"""', "", code, flags=re.DOTALL)  # funny usage of re
        code = re.sub(r"'''(.*?)'''", "", code, flags=re.DOTALL)

        print("Filtered comments")

        # print(code[:100])

        # filter non-ascii
        code = re.sub(r"[^\x00-\x7F]+", "", code.lower())
        # print(code[:100])
        print("Filtered non-ascii")

        #  # Handle hex/binary/octal sequences
        # def split_number_sequence(match):
        #     prefix, digits = match.group(1), match.group(2)
        #     return f"{prefix} " + " ".join(digits)

        # code = re.sub(r'(0x)([0-9a-f]+)', split_number_sequence, code)
        # code = re.sub(r'(0b)([01]+)', split_number_sequence, code)
        # code = re.sub(r'(0o)([0-7]+)', split_number_sequence, code)

        # print("Coped with hex")


        # each reserved word/symbol is a token. We split by space at the end, so this works.
        for word in self.reserved_keywords:
            code = re.sub(rf"\b{word}\b", f" {word} ", code)

        print("Reserved words")
        for symbol in self.symbols:
            code = code.replace(symbol, f" {symbol} ")

        print("Symbols")

        # print(code[:100])

        # Split identifiers by spaces, underscores, hyphens, or capitalization
        def split_token(token):
            result = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
            result = re.sub(r"([_-])", r" \1 ", result)
            return [part.lower() for part in result.split() if part.strip()]

        code = code.replace("	", " <TAB> ").replace("\n", " <NEWLINE> ")
        print("Tabs + newlines")
        

        tokens = []
        for token in code.split(" "):
            if token.strip():
                tokens.extend(split_token(token))

        print("Split tokens")
        token_freqs = {"<PAD>": 0}
        for token in [tok for tok in tokens if tok.strip()]:
            if token not in token_freqs:
                token_freqs[token] = 1
            else:
                token_freqs[token] += 1
        print("Counted freqs")
        # use cutoff thresh to replace tokens with UNK
        cutoff_thresh = self.cutoff_thresh
        if self.use_vocab_size_instead:
            print("Using vocab size instead")
            sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
            allowed_tokens = set(token for token, _ in sorted_tokens[:self.vocab_size-1]) # -1 for PAD
            for i in range(len(tokens)):
                if tokens[i] not in allowed_tokens and tokens[i] != "<PAD>":
                    print(f"Replacing token with UNK: {tokens[i]}")
                    tokens[i] = "<UNK>"

        else:
            cutoff_amt = np.percentile(list(token_freqs.values()), (1-cutoff_thresh) * 100)
            print(f"Cuttoff amount: {cutoff_amt} using threshold {cutoff_thresh}")
            for i, (token, freq) in enumerate(token_freqs.items()):
                if freq < cutoff_amt and token != "<PAD>":
                    print(f"Replacing token with UNK: {tokens[i]}")
                    tokens[i] = "<UNK>" # TODO: make sure this works



            

        print(tokens[500:700])

        print("500-700")

        return [tok for tok in tokens if tok.strip()]

    def encode(self, code):
        tokens = code.split(" ")
        ids = []

        for token in tokens:
            # New token
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
            ids.append(self.token_to_id[token])

        return ids

    def decode(self, ids):
        result = ""
        for id in ids.tolist():
            for token, id_iterator in self.token_to_id.items():
                if id_iterator == id:
                    result += token
                    result += " "

        return result
    
    def raw_decode(self, id: int):
        for token, id_iterator in self.token_to_id.items():
                if id_iterator == id:
                    return token

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

    def save_vocab(self, file_path):
        with open(file_path, "w") as file:
            for token, id in self.token_to_id.items():
                file.write(f"{token}\t{id}\n")

    def load_vocab(self, file_path):
        self.token_to_id = {}
        with open(file_path, "r") as file:
            for line in file.read().split("\n"):
                try:
                    token, id = line.strip().split("\t")
                    self.token_to_id[token] = int(id)
                except ValueError:
                    # print(line)
                    # print("^^ is error")
                    pass  # Should be fine, ends up being blank lines

    @staticmethod
    def attention_mask(encoded_sequence, mask_token_ids=[0]):
        mask_token_tensor = torch.tensor(mask_token_ids, dtype=torch.int)
        # print(mask_token_tensor)
        # print(encoded_sequence)
        return (encoded_sequence.unsqueeze(1) != mask_token_tensor).all(dim=1).int()


class DummySequentialDataManager:
    def __init__(self, root_dir, vocab_size=5000):
        print("init")
        self.root_dir = root_dir
        self.vocab_size = vocab_size
        with open(os.path.join(root_dir, "data/corpus_processed.txt"), "w+") as f:
            f.write("dummy")

    def encode(self, text: str):
        return [list(range(50))]

    def decode(self, ids):
        l = ids
        if isinstance(l, torch.Tensor):
            l = ids.tolist()
        if isinstance(l, int):
            l = [l]

        return " ".join([str(id) for id in l])

    @staticmethod
    def attention_mask(encoded_sequence, mask_token_ids=[]):
        mask_token_tensor = torch.tensor(mask_token_ids, dtype=torch.int).to(encoded_sequence.device)
        # print(mask_token_tensor)
        # print(encoded_sequence)
        return (encoded_sequence.unsqueeze(1) != mask_token_tensor).all(dim=1).int()


class TextCorpusDataset(Dataset):
    def __init__(
        self,
        root_dir="./test-data",
        train=False,
        max_length=512,
        vocab_size=10000,
        IS_DUMMY=False,
        IS_CODE=False,
        IS_CUSTOM=False,
        sliding_window=False,
        stride=1,
    ):
        print(root_dir)
        self.root = root_dir
        self.sliding_window = sliding_window
        self.window_size = max_length
        self.stride = stride

        if IS_DUMMY:
            self.manager = DummySequentialDataManager(root_dir=root_dir)
        elif IS_CODE:
            if IS_CUSTOM:
                self.manager = CodeCustomTokenizerManager(root_dir=root_dir)
            else:
                self.manager = CodeBPEModelManager(
                    root_dir=root_dir, vocab_size=vocab_size
                )
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
                encoded = self.manager.encode(text)

                self._chunk_and_save(encoded)

        end_t = time.time()
        print(f"Dataset loading took {end_t - start_t} seconds.")


        #TODO: more "optimization"
        self.chunks = self.chunks.to(DEVICE)
        self.dummy = torch.tensor([1],device=DEVICE)

    def _chunk_and_save(self, encoded):
        chunked_data = []
        if self.sliding_window:
            print("sliding!")
            for i in trange(0, len(encoded) - self.window_size + 1, self.stride, leave=False):
                chunked_data.append(torch.tensor(encoded[i : i + self.window_size], dtype=torch.int))
        else:
            for i in trange(0, len(encoded), self.max_length, leave=False):
                chunked_data.append(torch.tensor(encoded[i : i + self.max_length], dtype=torch.int))

            # me when the last item is not necessarily of length self.max_length
            padded_chunk = torch.zeros(self.max_length, dtype=torch.int)
            padded_chunk[: len(chunked_data[-1])] = chunked_data[-1]
            chunked_data[-1] = padded_chunk

        self.chunks = torch.stack(chunked_data)
        torch.save(self.chunks, self.cache_file)

    # unused
    # def _sliding_window(self, sequence, window_size, stride):
    #     windows = []
    #     for i in range(0, len(sequence) - window_size + 1, stride):
    #         windows.append(sequence[i : i + window_size])
    #     return torch.stack(windows)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx): #TODO: optimized, but change it back if it doesn't work
        seq = self.chunks[idx]
        return seq, self.dummy  #self.manager.attention_mask(seq)


# print("Running....")
dataset = TextCorpusDataset(
    root_dir=os.path.expanduser(
        # "./dummy-data-dir"
        # "./smaller-er-test-data"
        #"./smaller-test-data"
        #"~/torch_datasets/github-python/all_trains_subset_corpus"
        "~/torch_datasets/github-python/corpus"
    ),  # os.path.expanduser("~/torch_datasets/wikitext/train")
    vocab_size=2000,
    IS_CODE=True,  # Remember to change!
    IS_CUSTOM=True,
    # IS_DUMMY=True,
    max_length=256,
    sliding_window=True,
    stride=128
)
dset_size = int(len(dataset))
train_size = int(0.8 * dset_size)# int(dset_size - 2)
test_size = int(dset_size - train_size)
if test_size == 2:
    print("alert! test size is 2 or whatever. Change this back please.")

train_dataset, test_dataset, _ = random_split(
    dataset, [train_size, test_size, len(dataset) - train_size - test_size]
)

#train_dataset = dataset  # TODO change


def get_train_dataset():
    return train_dataset


def get_test_dataset():

    return test_dataset


def get_dataloader(dataset, batch_size=16):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    d = get_train_dataset()
    print("Number of samples: ", len(d))
    for a, b in d:
        # a, b = d[-1]
        manager = dataset.manager
        print(a)
        print(manager.decode(a))
        #print(a)
        print("--- sep batch --- ")

        print(f"Number of tokens used: {len(dataset.manager.token_to_id)}")
        break # lazy
