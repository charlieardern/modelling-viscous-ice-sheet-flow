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
    
class FourierNeuralField(nn.Module):
    def __init__(self, hidden_dim, N_x, N_t):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(2, hidden_dim),
        #     torch.nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.ReLU(),
        #     nn.Linear(hidden_dim, N_x+N_t)
        #     )
        self.N_x = N_x
        self.N_t = N_t
        alpha = torch.randn(N_x).reshape(-1,1,1,1)
        beta = torch.randn(N_t).reshape(-1,1,1,1)
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, X, s_init, L):
        k_x = (torch.arange(1, self.N_x+1).reshape(-1,1,1,1)-0.5)*torch.pi/L
        k_t = torch.arange(1, self.N_t+1).reshape(-1,1,1,1)
        X = X.unsqueeze(0)
        c = torch.cos(k_x*X[:,:,:,0].unsqueeze(-1))
        s = torch.sin(k_t*X[:,:,:,1].unsqueeze(-1))
        print(f"c shape: {c.shape}")
        print(f"s shape: {s.shape}")

        term_1 = torch.sum(self.alpha*c, dim=0)
        term_2 = torch.sum(self.beta*s, dim=0)
        return s_init +term_1*term_2

