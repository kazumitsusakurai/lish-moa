from torch.utils.data import Dataset
from torch import nn
import torch
import numpy as np


def add_noise(features, ratio=0.2):
    noise = np.random.randn(*features.shape) * ratio
    noisy_features = features + noise
    return noisy_features


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout_ratio0=None, dropout_ratio1=None):

        super(Encoder, self).__init__()

        self.norm0 = nn.BatchNorm1d(input_size)

        if dropout_ratio0 is not None:
            self.drop0 = nn.Dropout(dropout_ratio0)

        self.dense0 = nn.Linear(input_size, hidden_size)
        self.act0 = nn.ReLU()

        self.norm1 = nn.BatchNorm1d(hidden_size)

        if dropout_ratio1 is not None:
            self.drop1 = nn.Dropout(dropout_ratio1)

        self.dense1 = nn.Linear(hidden_size, output_size)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.norm0(x)

        if hasattr(self, 'drop0'):
            x = self.drop0(x)

        x = self.dense0(x)
        x = self.act0(x)

        x = self.norm1(x)

        if hasattr(self, 'drop1'):
            x = self.drop1(x)

        x = self.dense1(x)
        x = self.act1(x)

        return x


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout_ratio0=None, dropout_ratio1=None):
        super(Decoder, self).__init__()

        self.norm0 = nn.BatchNorm1d(input_size)

        if dropout_ratio0 is not None:
            self.drop0 = nn.Dropout(dropout_ratio0)

        self.dense0 = nn.Linear(input_size, hidden_size)
        self.act0 = nn.ReLU()

        self.norm1 = nn.BatchNorm1d(hidden_size)

        if dropout_ratio1 is not None:
            self.drop1 = nn.Dropout(dropout_ratio1)

        self.dense1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.norm0(x)

        if hasattr(self, 'drop0'):
            x = self.drop0(x)

        x = self.dense0(x)
        x = self.act0(x)

        x = self.norm1(x)

        if hasattr(self, 'drop1'):
            x = self.drop1(x)

        x = self.dense1(x)

        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,
                 dropout_ratio0=None, dropout_ratio1=None, dropout_ratio2=None, dropout_ratio3=None):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size1, output_size=hidden_size2,
                               dropout_ratio0=dropout_ratio0, dropout_ratio1=dropout_ratio1)

        self.decoder = Decoder(input_size=hidden_size2, hidden_size=hidden_size1, output_size=input_size,
                               dropout_ratio0=dropout_ratio2, dropout_ratio1=dropout_ratio3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class DaeDataset(Dataset):
    def __init__(self, data, feature_names, noise_ratio=0.2):
        self.feature_names = feature_names
        self.data = data.loc[:, feature_names].values
        self.noise_ratio = noise_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_fetures = self.data[idx]
        noisy_features = add_noise(self.data[idx], ratio=self.noise_ratio)
        return torch.FloatTensor(noisy_features), torch.FloatTensor(original_fetures)
