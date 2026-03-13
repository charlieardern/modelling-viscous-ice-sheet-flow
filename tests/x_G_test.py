import numpy as np
from solvers.solvers import numerical_fv_implicit
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

N = 100
x_G_0 = 0.1
nu = 80
num_steps = 100
num_microsteps = 500
t_final = 280
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
alpha = 0.5*np.sqrt(g_prime/g) # alpha required for A=1
A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

# initial state:
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

print("Computing x_G plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

timesteps = np.linspace(0, t_final, num=num_steps+1)
_, _, _ , x_G = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True, sheet_accumulation=False)

x_G = x_G/(2*B*C**4)
timesteps = timesteps*(a*L/(2*B**2*C**5))

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,5), dpi=300)
ax.set_xlabel(r"$t$ (dimensionless)", fontname="Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$x$ (dimensionless)", fontname="Latin Modern Roman", fontsize=14)
ax.set_title("Time evolution of grounding line position", fontname="Latin Modern Roman", fontsize=16)
ax.plot(timesteps, x_G.reshape(-1), c="blue", label=r"$x_G(t)$")
ax.set_ylim(np.min(x_G), np.max(x_G)+0.3*(np.max(x_G)-np.min(x_G)))
ax.set_xlim(0, np.max(timesteps))
x_G_steady = 1/(A+eps)
ax.plot([timesteps[0], timesteps[-1]], [x_G_steady, x_G_steady], alpha=0.2, c="blue", linestyle="--", label="Analytical \nsteady state")
plt.legend()

plt.savefig(folder + "x_G_plot.png")

print("Complete.")