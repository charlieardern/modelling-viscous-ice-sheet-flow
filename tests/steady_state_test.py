import numpy as np
from solvers.solvers import numerical_fv
import matplotlib.pyplot as plt
import os

N = 100
x_G_0 = 0.3
nu = 80
num_steps = 100
num_microsteps = 12000
t_final = 400
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
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

print("Computing steady state plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

x, h, _, _ = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, sheet_accumulation=False)

# Non-dimensionalise x and h:
x_pred = x[:,-1]/(2*B*C**4)
h_pred = h[:,-1]/(B*C)

plt.figure(figsize=(8,6), dpi=300)

plt.plot(x_pred, 1-x_pred, label="analytical")
plt.plot(x_pred, h_pred, label="numerical")
plt.legend()
plt.title("Steady state")
plt.savefig(folder + "steady_state_plot.png")
plt.close()

plt.figure(figsize=(8,5), dpi=300)
plt.plot(x_pred, 1-x_pred-h_pred)
plt.title("(analytical - numerical) for steady state", fontname="Latin Modern Roman", fontsize=16)
plt.xlabel(r"$\hat{x}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$\hat{z}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.savefig(folder + "steady_state_diff_plot.png")

print("Complete.")