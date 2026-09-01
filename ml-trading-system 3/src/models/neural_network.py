"""
Neural network component of the ensemble.

TradingNet is a feedforward network with residual blocks for depth and
a single self-attention block applied across the feature dimension of
a single day's feature vector. It does not process a sequence of time
steps and has no positional encoding, so it is not a sequence
Transformer in the GPT/BERT sense.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AttentionBlock(nn.Module):
    """Single self-attention block operating over the feature dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ V


class ResidualBlock(nn.Module):
    """Residual (skip-connection) block: helps gradient flow in deeper nets."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class TradingNet(nn.Module):
    """Feedforward network with residual blocks and one attention layer."""

    def __init__(self, input_dim: int, hidden_dims=(512, 256, 128), dropout: float = 0.3):
        super().__init__()
        hidden_dims = list(hidden_dims)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attention = AttentionBlock(hidden_dims[0])

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout) for _ in range(2)
        ])

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.feature_net = nn.Sequential(*layers)

        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.attention(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.feature_net(x)
        return self.output(x)


# Alias for backwards compatibility with earlier code that referenced
# this class as TradingTransformer.
TradingTransformer = TradingNet


class TradingDataset(Dataset):
    """Simple (X, y) tensor dataset for the DataLoader."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
