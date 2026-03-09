import numpy as np
from solvers.solvers import numerical_fv
import matplotlib.pyplot as plt
import os

N = 100
x_G_0 = 0.3
alpha = 0.24
nu = 80
num_steps = 100
num_microsteps = 12000
t_final = 200
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)

# initial state:
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

print("Computing x_G plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

timesteps = np.linspace(0, t_final, num=num_steps+1)
_ , x_G = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True, sheet_accumulation=False)

x_G = x_G/(2*B*C**4)

plt.figure(figsize=(7,5), dpi=300)
plt.xlabel(r"$t$ (s)", fontname="Latin Modern Roman", fontsize=14)
plt.ylabel(r"$x_G$ (m)", fontname="Latin Modern Roman", fontsize=14)
plt.title("Time evolution of grounding line position", fontname="Latin Modern Roman", fontsize=16)
plt.plot(timesteps, x_G.reshape(-1))
plt.ylim(np.min(x_G), np.max(x_G)+0.1*(np.max(x_G)-np.min(x_G)))

plt.savefig(folder + "x_G_plot.png")

print("Complete.")