import torch

from transformers import AutoModel, AutoTokenizer


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

    def forward(self, x):
        pass

    def forward_no_grad(self, x):
        pass
