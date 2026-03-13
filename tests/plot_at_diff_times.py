import numpy as np
from solvers.solvers import numerical_fv_implicit
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

N = 100
x_G_0 = 0.1
alpha = 0.24
nu = 80
num_steps = 100
num_microsteps = 500
t_final = 280
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150
domain_width = 5

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho
D = 2*B*C**4/L

print(f"A: {A}, D: {D}, ε: {eps}")


print(f"scaled final time: {t_final*(a*L/(2*B**2*C**5))}")

# initial state:
h_0 = 0.7*(1-np.linspace(0,1,num=N)+0.002+0.1)

print("Producing plot at different times...")
x, h, x_shelf, H_shelf = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L)

# Non-dimensionalise x and h:
x = x/(2*B*C**4)
h = h/(B*C)
x_shelf = x_shelf/(2*B*C**4)
H_shelf = H_shelf/(B*C)

plt.figure(figsize=(10,4.8), dpi=400)
plt.title("Time evolution of ice sheet and shelf", fontname = "Latin Modern Roman", fontsize=16)
plt.axis('equal')
plt.xlim(0, domain_width)
plt.ylim(-1.1, 1.1)
plt.xlabel(r"$\hat{x}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$\hat{z}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.plot([0,domain_width],[0,-A*domain_width], c='black', label='bedrock')
plt.plot([0,domain_width], [0,0], c="blue", alpha=0.2, label='water line')

frames_to_plot = [0, 10, 18, 40, 58, 99]

for i in tqdm(range(h.shape[1]-1)):
    #if (i%25 == 0) or (i == h.shape[1]-2):
    if i in frames_to_plot:
        h_above_water = h[-1,i]
        plt.plot(x[:,i], h[:,i], c='blue', linestyle="solid", label="sheet")
        plt.plot(x_shelf[:,i], -H_shelf[:,i]+h_above_water, c='blue', linestyle='solid', label='shelf')
        plt.plot([x_shelf[0,i], x_shelf[0,i]], [-H_shelf[0,i]+h_above_water, h_above_water], c='blue')
        plt.plot([x_shelf[-1,i], x_shelf[-1,i]], [-A*x[-1,i], -H_shelf[-1,i]+h_above_water], c='blue')
        plt.plot([x_shelf[-1,i], x_shelf[0,i]], [h_above_water,h_above_water], c='blue')
        print(f"Shape at dimensionless time: {(a*L/(2*B**2*C**5))*i*t_final/num_steps}")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

plt.ylim(-1.1, 1.1)
plt.savefig(folder + "plot_at_diff_times.png")
print("Complete.")