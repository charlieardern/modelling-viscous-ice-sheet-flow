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
    
class FourierField(nn.Module):
    def __init__(self, hidden_dim, N_x, N_t, device):
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
            nn.Linear(hidden_dim, N_x+N_t)
            )
        self.device = device
        self.N_x = N_x
        self.N_t = N_t


    def forward(self, X, s_init, L):
        coeffs = self.mlp(X) # (dim, num_t_steps, N_x+N_t)
        alpha = coeffs[:,:,:self.N_x] # (dim, num_t_steps, N_x)
        beta = coeffs[:,:,self.N_x:]# (dim, num_t_steps, N_t)
        print(f"alpha shape: {alpha.shape}")
        print(f"beta shape: {beta.shape}")

        k_x = (torch.arange(1, self.N_x+1).reshape(1,1,-1).to(self.device)-0.5)*torch.pi/L # (1, 1, N_x)
        k_t = torch.arange(1, self.N_t+1).reshape(1,1,-1).to(self.device) # (1, 1, N_t)
        c = torch.cos(k_x*X[:,:,0].unsqueeze(-1))
        s = torch.sin(k_t*X[:,:,1].unsqueeze(-1))
        print(f"c shape: {c.shape}")
        print(f"s shape: {s.shape}")

        term_1 = torch.sum(alpha*c, dim=-1).unsqueeze(-1)
        term_2 = torch.sum(beta*s, dim=-1).unsqueeze(-1)
        print(f"term_1.shape: {term_1.shape}")
        print(f"term_2.shape: {term_2.shape}")
        return s_init +term_1*term_2

class BetterFourierModel(nn.Module):
    def __init__(self, hidden_dim, N_x, N_t, device):
        super().__init__()
        A = torch.randn(N_x,N_t).reshape(1,1,N_x,N_t)
        self.A = nn.Parameter(A)
        self.device = device
        self.N_x = N_x
        self.N_t = N_t


    def forward(self, X, s_init, L):
        k_x = (torch.arange(1, self.N_x+1).reshape(1,1,-1).to(self.device)-0.5)*torch.pi/L # (1, 1, N_x)
        k_t = torch.arange(1, self.N_t+1).reshape(1,1,-1).to(self.device) # (1, 1, N_t)
        c = torch.cos(k_x*X[:,:,0].unsqueeze(-1))
        s = torch.sin(k_t*X[:,:,1].unsqueeze(-1)/100)
        c = c.reshape(c.shape[0], c.shape[1], c.shape[2], 1)
        s = s.reshape(s.shape[0], s.shape[1], 1, s.shape[2])
        out = torch.sum(s*c*self.A,dim=(-1,-2)).unsqueeze(-1)
        return s_init + out
    
class MixedBasisModel(nn.Module):
    def __init__(self, hidden_dim, N_x, N_t, device):
        super().__init__()
        A = torch.randn(N_x-1,N_t).reshape(1,1,N_x-1,N_t)
        self.A = nn.Parameter(A)
        self.device = device
        self.N_x = N_x
        self.N_t = N_t

    def forward(self, X, s_init, L):
        # Normalise x values:
        x_normed = X[:,:,0].unsqueeze(-1)/L
        t_normed = X[:,:,1].unsqueeze(-1)/100

        k_x = torch.arange(2, self.N_x+1).reshape(1,1,-1).to(self.device) # (1, 1, N_x-1)
        k_t = torch.arange(1, self.N_t+1).reshape(1,1,-1).to(self.device) # (1, 1, N_t)

        #x_term = X[:,:,0].unsqueeze(-1)**k_x-L**k_x
        x_term = x_normed**k_x-1
        
        #print(x_term)
        s = torch.sin(k_t*t_normed)
        s = s.reshape(s.shape[0], s.shape[1], 1, s.shape[2])
        x_term = x_term.reshape(x_term.shape[0], x_term.shape[1], x_term.shape[2], 1)
        out = torch.sum(s*x_term*self.A,dim=(-1,-2)).unsqueeze(-1)
        #print(torch.max(out))
        return s_init + out