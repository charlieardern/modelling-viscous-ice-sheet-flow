import numpy as np
from solvers.solvers import numerical_fv
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

N = 100
x_G_0 = 0.7
alpha = 0.16
nu = 250
num_steps = 100
num_microsteps = 12000
t_final = 200
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 80

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)

# initial state:
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

print("Computing solution...")
converge, _ = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test=True)

timesteps = np.linspace(1e-4, 1e-2, num=10)
chisteps = np.linspace(0.002, 0.02, num=10)

plot_t_steps = (a*L/(2*B**2*C**5))**timesteps

for t_idx in tqdm(range(len(timesteps))):
    for chi_idx in range(len(chisteps)):
        delta_t = timesteps[t_idx]
        delta_chi = timesteps[chi_idx]
        h_0 = 1-np.linspace(0,1,num=round(1/delta_chi))+0.002+0.1
        num_microsteps = t_final/(num_steps*delta_t)
        
        


folder = "figures/"
os.makedirs(folder, exist_ok=True)

print("Complete.")