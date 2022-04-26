import os
import torch
import jsonlines

from tqdm import tqdm
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, root_dir="data/dev.jsonl", tokenizer=None):
        super(ReviewDataset, self).__init__()
        
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.text, self.tensor, self.label = [], [], []
        
    def load(self):
        with jsonlines.open(self.root_dir) as freader:
            for line in tqdm(freader.iter(), desc=f"Loading from {self.root_dir}"):
                self.text.append(line["text"])
                self.tensor.append(torch.tensor([self.tokenizer.encode(line["text"])]))
                self.label.append(line["labels"])

        return self
        
    def __getitem__(self, x):
        return self.text[x], self.label[x]

    def __len__(self):
        assert len(self.text) == len(self.label)
        return len(self.text)










