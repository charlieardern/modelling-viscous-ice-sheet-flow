import numpy as np
from solvers.solvers import IceSheetSolver
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

print("Computing steady state plot...")

N = 100
x_G_0 = 0.1
nu = 80
num_microsteps = 500
t_final = 2
L = 150
D = 0

# initial state:
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N))

folder = "figures/"
os.makedirs(folder, exist_ok=True)

x_bed = np.linspace(0,20, num=2000)
b_bed = x_bed

system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)
system.compute_solution(compute_shelf=False)
x, h = system.x_tnsr, system.h_tnsr

times_to_plot = [3, 10, 25, 99]
alphas = [0.3, 0.5, 0.75, 1]

fig, ax = plt.subplots(constrained_layout=True, figsize=(4.8,3.2), dpi=400)

for i in range(len(times_to_plot)):
    label = f"t = {(times_to_plot[i]*t_final/system.num_steps):.1f}"
    h_pred = h[:,times_to_plot[i]]
    x_pred = x[:,times_to_plot[i]]
    ax.plot(x_pred, h_pred, label=label, c="orange", alpha=alphas[i], linewidth=2)

ax.plot(x[:,-1], (1-x[:,-1]), label="Analytical \nsteady state", c="blue", alpha=1, linestyle="--", linewidth=1)
plt.legend(prop={"family":"Latin Modern Roman", "size":12})
ax.set_xlabel(r"$x$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_title("Time evolution of sheet", fontname="Latin Modern Roman", fontsize=14)
plt.savefig(folder + "steady_state_plot.png")
plt.close()

print("Complete.")