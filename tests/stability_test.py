import numpy as np
from solvers.solvers import numerical_fv
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

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

print("Computing solution...")


# timesteps = np.linspace(1e-4, 1e-2, num=10)
# chisteps = np.linspace(0.002, 0.02, num=10)

timesteps = np.linspace(1e-3, 1e-4, num=10)
chisteps = np.linspace(0.008, 0.016, num=10)

scaled_t_steps = (a*L/(2*B**2*C**5))**timesteps

folder = "figures/"
os.makedirs(folder, exist_ok=True)

plt.figure(dpi=200)
plt.xlabel(r"$\Delta \chi$")
plt.ylabel(r"$\Delta t$ (dimensionless)")
plt.title("Solver stability")
chi_range = max(chisteps)-min(chisteps)
t_range = max(chisteps)-min(chisteps)
plt.xlim(min(chisteps)-0.1*chi_range, max(chisteps)+0.1*chi_range)
plt.ylim(min(scaled_t_steps)-0.1*t_range, max(scaled_t_steps)+0.1*t_range)

boundary_cross = False

for t_idx in tqdm(range(len(timesteps))):
    boundary_crossed = False
    for chi_idx in range(len(chisteps)):
        delta_t = timesteps[t_idx]
        delta_chi = chisteps[chi_idx]
        h_0 = 1-np.linspace(0,1,num=round(1/delta_chi))+0.002+0.1
        num_microsteps = round(t_final/(num_steps*delta_t))
        
        print(f"N = {round(1/delta_chi)} | num_microsteps = {num_microsteps}")
        if boundary_crossed:
            plt.scatter(delta_chi, scaled_t_steps[t_idx], c="green")
        else:
            converge, _ = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
            colour = "green" if converge else "red"
            plt.scatter(delta_chi, scaled_t_steps[t_idx], c=colour)
            boundary_crossed = converge
        plt.savefig(folder+ "stability_plot.png")

print("Complete.")