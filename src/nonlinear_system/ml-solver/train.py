import torch
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer

from models import BetterFourierModel

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
    c_1 = 0
    c_2 = -0.1
    return c_1*(x**3-L**3)+c_2*(x**2-L**2)+s_G
s_t0 = s_0(x[:,0,0].reshape(dim,1,1), s_G).detach()


# Spacetime grid of points at which to evaluate model:
X = torch.concatenate([x,t], dim=2)

# Bedrock at spacetime points
bed = torch.zeros(dim, requires_grad=True).reshape(dim,1,1).repeat(1,num_t_steps,1).to(device)

model = BetterFourierModel(hidden_dim=64, N_x=1, N_t=10, device=device).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

train_time_start = default_timer()
epochs = 1000
losses = []

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch+1}\n-----------")
    model.train()

    s_pred = model(X, s_t0, L)
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

    f_err = deriv_1[:,:,1]-(g/(3*eta))*deriv_2[:,:,0]-a
    loss = torch.sum(f_err**2)
    mse = torch.sqrt(loss/(num_t_steps*dim))
    print(f"mse: {mse}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

model.eval()
s = model(X, s_t0, L).detach().cpu().numpy()
plt.plot(losses)
plt.savefig("figures/loss_curves.png")
plt.close()
print(s.shape)

np.save("saved_objects/ml_solution.npy", s)

plt.figure()
plt.plot(x[:,0,0].detach().cpu().numpy(), s[:,0,0])
plt.savefig("figures/t_init.png")