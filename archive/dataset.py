import torchvision.datasets as dset
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import glob
import os
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split





class GithubDataset(Dataset):
    def __init__(self, root_dir=os.path.expanduser("~/torch_datasets/github-python/corpus"), train=False, max_length=512):
        self.root = root_dir
        self.file_list = glob.glob(os.path.join(root_dir, '*.*'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        path = self.file_list[idx]

        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
            code = file.read()


        encoding = self.tokenizer(code, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        #print(encoding.keys)

        return input_ids, attention_mask
    
        
dataset = GithubDataset()#root_dir="./test-data/")
dataset = GithubDataset(root_dir="./test-data/")
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
    t = AutoTokenizer.from_pretrained('bert-base-uncased')
    for i in a:
        print(t.decode(i.item()), end=" ")
    print()