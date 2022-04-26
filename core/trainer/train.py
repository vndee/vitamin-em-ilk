import torch
import argparse

from core.trainer.loader import ReviewDataset
from transformers import AutoTokenizer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    dataset = ReviewDataset(tokenizer=tokenizer).load()
