import numpy as np
from solvers.solvers import numerical_fv_implicit
import matplotlib.pyplot as plt
import os

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
print(alpha)

A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

# initial state:
h_0 = 0.7*(1-np.linspace(0,1,num=N)+0.002+0.1)

print("Computing steady state plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

x, h, _, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, sheet_accumulation=False)

# Non-dimensionalise x and h:
x_pred = x[:,-1]/(2*B*C**4)

times_to_plot = [5, 10, 20, 99]
alphas = [0.25, 0.5, 0.75, 1]

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,5), dpi=300)


for i in range(len(times_to_plot)):
    label = f"t = {((a*L/(2*B**2*C**5))*times_to_plot[i]*t_final/num_steps):.1f}"
    h_pred = h[:,times_to_plot[i]]/(B*C)
    ax.plot(x_pred, h_pred, label=label, c="orange", alpha=alphas[i], linewidth=2)

ax.plot(x_pred, (1-x_pred), label="Analytical \nsteady state", c="blue", alpha=0.6, linestyle="--", linewidth=1)
plt.legend(prop={"family":"Latin Modern Roman", "size":12})
ax.set_xlabel(r"$x$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$z$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_title("Time evolution of sheet compared with steady state", fontname="Latin Modern Roman", fontsize=16)
plt.savefig(folder + "steady_state_plot.png")
plt.close()

print("Complete.")