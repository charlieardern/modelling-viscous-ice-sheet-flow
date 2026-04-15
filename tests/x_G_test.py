import numpy as np
from solvers.solvers import IceSheetSolver
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

print("Computing x_G plot...")

N = 100
x_G_0 = 0.1
nu = 80
num_microsteps = 500
t_final = 2
L = 150
D = 0
A = 1

# initial state:
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N))

folder = "figures/"
os.makedirs(folder, exist_ok=True)

x_bed = np.linspace(0,20, num=2000)
b_bed = A*x_bed

system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)
system.compute_solution(compute_shelf=False)
x_G = system.x_G_tnsr

timesteps = np.linspace(0, t_final, num=system.num_steps+1)
x_G = x_G.reshape(-1)

fig, ax = plt.subplots(constrained_layout=True, figsize=(5.4,3.6), dpi=300)
ax.set_xlabel(r"$t$ (dimensionless)", fontname="Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$x_G$ (dimensionless)", fontname="Latin Modern Roman", fontsize=14)
ax.set_title("Time evolution of grounding line position", fontname="Latin Modern Roman", fontsize=16)
ax.plot(timesteps, x_G.reshape(-1), c="blue", label=r"$x_G(t)$")
ax.set_ylim(np.min(x_G), np.max(x_G)+0.1*(np.max(x_G)-np.min(x_G)))
ax.set_xlim(0, np.max(timesteps))
x_G_steady = 1/(A+np.interp(x_G[-1], x_bed, b_bed)*system.rho_w*system.g_prime*A/(system.rho*system.g))
ax.plot([timesteps[0], timesteps[-1]], [x_G_steady, x_G_steady], alpha=0.2, c="blue", linestyle="--", label="Analytical \nsteady state")
plt.legend(fontsize=14)

plt.savefig(folder + "x_G_plot.png")

print("Complete.")