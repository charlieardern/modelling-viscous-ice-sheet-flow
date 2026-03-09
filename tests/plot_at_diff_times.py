import numpy as np
from solvers.solvers import numerical_fv
import os
import matplotlib.pyplot as plt
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
domain_width = 4

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)

# initial state:
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

print("Producing plot at different times...")
x, h, x_shelf, H_shelf = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L)

# Non-dimensionalise x and h:
x = x/(2*B*C**4)
h = h/(B*C)
x_shelf = x_shelf/(2*B*C**4)
H_shelf = H_shelf/(B*C)

plt.figure(figsize=(10,5), dpi=400)
plt.title("Time evolution of ice sheet and shelf", fontname = "Latin Modern Roman", fontsize=16)
plt.xlim(0, domain_width)
plt.ylim(-1.5, 1.5)
plt.xlabel(r"$\hat{x}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$\hat{z}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.plot([0,domain_width],[0,-A*domain_width], c='black', label='bedrock')
plt.plot([0,domain_width], [0,0], c="blue", alpha=0.2, label='water line')

for i in tqdm(range(h.shape[1]-1)):
    if (i%20 == 0) or (i == h.shape[1]-2):
        h_above_water = h[-1,i]
        plt.plot(x[:,i], h[:,i], c='blue', linestyle="solid", label="sheet")
        plt.plot(x_shelf[:,i], -H_shelf[:,i]+h_above_water, c='blue', linestyle='solid', label='shelf')
        plt.plot([x_shelf[0,i], x_shelf[0,i]], [-H_shelf[0,i]+h_above_water, h_above_water], c='blue')
        plt.plot([x_shelf[-1,i], x_shelf[-1,i]], [-A*x[-1,i], -H_shelf[-1,i]+h_above_water], c='blue')
        plt.plot([x_shelf[-1,i], x_shelf[0,i]], [h_above_water,h_above_water], c='blue')

folder = "figures/"
os.makedirs(folder, exist_ok=True)

plt.savefig(folder + "plot_at_diff_times.png")
print("Complete.")