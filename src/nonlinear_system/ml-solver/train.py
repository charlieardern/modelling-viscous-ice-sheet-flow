import torch
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer

from models import FeatureEngineeredNeuralField

from tqdm.auto import tqdm

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device="cpu"

print(f"Training on device: {device}")

# Set up physical system
s_G = 2.5
L = 4
a = 0.5
num_t_steps = 100
t_final = 10
dim = 200
w = L/dim
g = 9.81
eta = 20

x = torch.linspace(0+0.5*w,L-0.5*w, steps=dim, requires_grad=True).reshape(-1, 1, 1).repeat(1,num_t_steps,1).to(device)
t = torch.linspace(0,t_final, steps=num_t_steps, requires_grad=True).reshape(1,-1, 1).repeat(dim, 1, 1).to(device)
def s_0(x, s_G):
    return -s_G*x/L + 2*s_G
s_t0 = s_0(x[:,0,0].reshape(dim,1,1), s_G).detach()

# Spacetime grid of points at which to evaluate model:
X = torch.concatenate([x,t], dim=2)

# Bedrock at spacetime points
bed = torch.zeros(dim, requires_grad=True).reshape(dim,1,1).repeat(1,num_t_steps,1).to(device)

model = FeatureEngineeredNeuralField(hidden_dim=256).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

train_time_start = default_timer()
epochs = 400
losses = []

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1}\n-----------")
    model.train()

    s_pred = model(X,L, s_G, s_t0)
    deriv_1 = torch.autograd.grad(
        outputs=s_pred,
        inputs=X,
        create_graph=True,
        retain_graph=True,
        grad_outputs = torch.ones_like(s_pred)
    )[0]
    F = (s_pred-bed)**3*deriv_1[:,:,0].reshape(dim, num_t_steps, 1)
    deriv_2 = torch.autograd.grad(
        outputs=F,
        inputs=X,
        create_graph=True,
        retain_graph=True,
        grad_outputs = torch.ones_like(F),
    )[0]

    # Both of shape (1, num_t_steps, 2):
    boundary_0 = torch.concatenate([torch.zeros(num_t_steps).reshape(1,num_t_steps,1).to(device),t[0].reshape(1,num_t_steps,1)],axis=2)
    #boundary_L = torch.concatenate([L*torch.ones(num_t_steps).reshape(1,num_t_steps,1),t[0].reshape(1,num_t_steps,1)],axis=2)
    #print(f"boundary_L.shape: {boundary_L.shape}")

    #boundary_L_err = 0.01*torch.sum((model(boundary_L)-s_G)**2)
    boundary_0_values = model(boundary_0, L, s_G, s_0(torch.zeros(1).to(device), s_G).reshape(1,1,1))
    boundary_0_deriv = torch.autograd.grad(
        outputs=boundary_0_values,
        inputs=boundary_0,
        create_graph=True,
        retain_graph=True,
        grad_outputs = torch.ones_like(boundary_0_values),
    )[0]

    boundary_0_err = 0.5*torch.sum(boundary_0_deriv[:,:,0]**2)

    init_X = torch.concatenate([x[:,0,0].reshape(dim,1,1),torch.zeros_like(s_t0)],dim=2)

    #s0_pred = model(init_X, L, s_G, s_t0)
    #s_0_mse = 0.01*torch.sum((s0_pred-s_t0)**2)

    f_err = deriv_1[:,:,1]-(g/(3*eta))*deriv_2[:,:,0]-a
    mse_f = 0.001*torch.sum(f_err**2)
    print(f"mse_f: {mse_f}")
    print(f"boundary_0_err: {boundary_0_err}")
    #print(f"boundary_L_err: {boundary_L_err}")
    #print(f"s_0_mse: {s_0_mse}")
    loss = mse_f + boundary_0_err #+ s_0_mse boundary_L_err + 
    print(f"loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

model.eval()
s = model(X, L, s_G, s_t0).detach().cpu().numpy()
plt.plot(losses)
plt.savefig("figures/loss_curves.png")

print(s.shape)

np.save("saved_objects/ml_solution.npy", s)