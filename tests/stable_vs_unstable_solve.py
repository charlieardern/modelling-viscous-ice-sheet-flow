import numpy as np
from solvers.solvers import numerical_fv_implicit_general_b
import matplotlib.pyplot as plt
import os

N = 501
x_G_0 = 0.1
nu = 80
num_steps = 100
num_microsteps = 6
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

B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

# initial state:
h_0 = 0.7*(1-np.linspace(0,1,num=N)+0.002+0.1)

print("Computing steady state plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

x_bed = np.linspace(0,20, num=2000)
b_bed = x_bed

x, h, _, _ = numerical_fv_implicit_general_b(h_0, x_bed, b_bed, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, a, L, sheet_accumulation=False, test_mode=True)

print(f"x.shape: {x.shape}")
print(f"h.shape: {h.shape}")

# Non-dimensionalise x and h:
x_pred = x[:,-1]/(2*B*C**4)

times_to_plot = [5, 10, 20, 99]
alphas = [0.25, 0.5, 0.75, 1]

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,3.5), dpi=400)

ax.plot(x_pred, h[:,-1]/(B*C))

#ax.plot(x_pred, (1-x_pred), label="Analytical \nsteady state", c="blue", alpha=0.6, linestyle="--", linewidth=1)
plt.legend(prop={"family":"Latin Modern Roman", "size":12})
ax.set_xlabel(r"$x$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$z$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_title("Time evolution of sheet compared with steady state", fontname="Latin Modern Roman", fontsize=16)
#ax.set_ylim(0, 1.3)
ax.set_xlim(0, 1)
plt.savefig(folder + "stable_vs_unstable.png")
plt.close()

print("Complete.")