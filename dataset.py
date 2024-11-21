import torchvision.datasets as dset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import glob
import os
from torch.utils.data import Dataset, DataLoader, random_split
from shutil import copyfile

import youtokentome as yttm



class BPEModelManager:
    def __init__(self, root_dir, vocab_size=5000):
        model_path = os.path.join(root_dir, "bpe_model.model")
        try:

            bpe = yttm.BPE(model=model_path)
            if bpe.vocab_size() != vocab_size:
                print("Vocab size didn't match, assuming bad model.")
                print(f"Expected size {vocab_size}, got {bpe.vocab_size()}")
                copyfile(model_path, os.path.join(root_dir, "bpe_model.model.old"))
                raise ValueError
            
        except ValueError:

            yttm.BPE.train(data=os.path.join(root_dir, "data/corpus.txt"), vocab_size=vocab_size, model=model_path)
            bpe = yttm.BPE(model=model_path)

        self.bpe = bpe
    def encode(self, text: str):
        return self.bpe.encode([text], output_type=yttm.OutputType.ID)

    def decode(self, ids):
        return self.bpe.decode(ids)
    
    @staticmethod
    def test():
        manager = BPEModelManager("test-data")
        print(manager.encode("This is a test"))
        print(manager.decode(manager.encode("This is a test")))

    @staticmethod
    def attention_mask(encoded_sequence, mask_token_ids=[0,1,2,3]):
        return [1 if token not in mask_token_ids else 0 for token in encoded_sequence]

class TextCorpusDataset(Dataset):
    def __init__(self, root_dir=os.path.expanduser("~/torch_datasets/github-python/corpus"), train=False, max_length=512, vocab_size=10000):
        self.root = root_dir
        self.manager = BPEModelManager(root_dir=root_dir,vocab_size=vocab_size)
        self.max_length = max_length

        with open(os.path.join(root_dir, "data/corpus.txt"), "r") as file:
            text = file.read()
            encoded = self.manager.encode(text)[0]
            self.chunks = [encoded[i:i+self.max_length] for i in range(0, len(encoded), self.max_length)]




    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        seq = self.chunks[idx]

        return seq, self.manager.attention_mask(seq) #todo: convert to tensor.


        
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
    print('Number of samples: ', len(d))

    a,b = d[4]
    manager = BPEModelManager("./test-data/", vocab_size=10000)
    print(manager.decode(a))
    print()
