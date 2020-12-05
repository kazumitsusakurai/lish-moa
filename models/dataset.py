

import torch
from torch.utils.data import Dataset


class MoaDataset(Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.targets is None:
            return torch.FloatTensor(self.data[idx]), 0

        return (
            torch.FloatTensor(self.data[idx]),
            torch.FloatTensor(self.targets[idx]),
        )
