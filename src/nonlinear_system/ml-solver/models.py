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
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            )

    def forward(self, x, L, s_G, s_init):
        bc_L_func = L-x[:,:,0].unsqueeze(-1)
        ic_func = (x[:,:,1].unsqueeze(-1))
        print(f"s_init.shape: {s_init.shape}")
        print(f"self.mlp(x)*bc_L_func*ic_func.shape: {(self.mlp(x)*bc_L_func*ic_func).shape}")
        return self.mlp(x)*bc_L_func*ic_func+s_init

class FeatureEngineeredNeuralField(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            )

    def forward(self, x, L, s_G, s_init):
        X = torch.concatenate([x,x**2, (x[:,:,0]*x[:,:,1]).unsqueeze(-1)],dim=-1)
        s = self.mlp(X)
        bc_L_func = L-x[:,:,0].unsqueeze(-1)
        ic_func = (x[:,:,1].unsqueeze(-1))
        print(f"s_init.shape: {s_init.shape}")
        print(f"self.mlp(x)*bc_L_func*ic_func.shape: {(s*bc_L_func*ic_func).shape}")
        return s*bc_L_func*ic_func+s_init