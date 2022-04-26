import unittest

from transformers import AutoTokenizer
from torch.utils.data import Dataset
from core.trainer.loader import ReviewDataset


class TestLoader(unittest.TestCase):
    def test_init(self):
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        dataset = ReviewDataset(tokenizer=tokenizer).load()
        self.assertIsInstance(dataset, Dataset)

    def test_access(self):
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        dataset = ReviewDataset(tokenizer=tokenizer).load()
        self.assertIsNotNone(dataset[0])
        self.assertIsNotNone(dataset[len(dataset) - 1])


if __name__ == "__main__":
    unittest.main()

