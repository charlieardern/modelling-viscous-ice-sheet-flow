import torch

from torch import nn

class SimpleNeuralField(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            )
    def forward(self, x):
        return self.mlp(x)