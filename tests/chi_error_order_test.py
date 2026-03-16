import numpy as np
from solvers.solvers import numerical_fv_implicit
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def linear(x, a, b):
    return a*x + b

compute_fine_space = False
compute_rmse = False

x_G_0 = 0.3
nu = 80
num_steps = 100
t_final = 100
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
alpha = 0.24
print(alpha)

A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

print("Computing error order tests...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

# Run test for delta_chi order -------------------------------------------
num_microsteps = 3000

print(f"timestep used: {(a*L/(2*B**2*C**5))*t_final/(num_steps*num_microsteps)}")

#N_values = np.round(1/np.linspace(1/4, 1/100, num=30))
N_values = [3, 4, 5, 6, 7, 8, 10, 20, 40, 60, 80]
#N_values = np.round(np.linspace(5,100, num=30))
N_fine = 700
j = np.linspace(0,N_fine-1, num=N_fine)
chi_fine = (j+0.5)/N_fine

if compute_fine_space:
    h_0 = 1-np.linspace(0,1,num=N_fine)+0.002+0.1
    _, h_fine, _, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
    h_fine = h_fine[:,-1]/(B*C)
    os.makedirs("saved_objects/", exist_ok=True)
    np.save("saved_objects/fine_solution.npy", h_fine)
else:
    h_fine = np.load("saved_objects/fine_solution.npy")

rmse_list = []

if compute_rmse:
    for i in tqdm(range(len(N_values))):
        N = int(N_values[i])
        j = np.linspace(0,N-1, num=N)
        chi_new = (j+0.5)/N

        h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
        x, h, _, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
        h_pred = h[:,-1]/(B*C)
        h_fine_i = np.interp(chi_new, chi_fine, h_fine)

        rmse_list.append(np.sqrt(np.mean((h_pred-h_fine_i)**2)))
        print(f"rmse: {rmse_list[i]}")
    np.save("saved_objects/rmse_chi.npy", rmse_list)
else:
    rmse_list = np.load("saved_objects/rmse_chi.npy")

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)

ax.set_title("RMSE against fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"RMSE$_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
popt, pcov = curve_fit(quadratic, 1/np.array(N_values), rmse_list)
#ax.plot(1/np.array(N_values), quadratic(1/np.array(N_values), *popt), c="red")
ax.plot(np.linspace(1/N_values[-1],1/N_values[0], num=200), quadratic(np.linspace(1/N_values[-1],1/N_values[0], num=200), *popt), c="red", label="Quadratic fit")
ax.scatter(1/np.array(N_values), rmse_list, c="black")
ax.legend()

plt.savefig(folder + "rmse_vs_dchi.png")
plt.close()

print(f"Spatial quadratic weight: {popt[0]} +- {np.sqrt(pcov[0,0])}")
print(f"Spatial linear weight: {popt[1]} +- {np.sqrt(pcov[1,1])}")

# Log plots

log_chi = np.log(1/np.array(N_values))
log_rmse = np.log(rmse_list)

popt, pcov = curve_fit(linear, log_chi, log_rmse)

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)

ax.plot(log_chi, linear(log_chi, *popt), label=f"Linear fit with \nm = {popt[0]:.2f}", c="red")
ax.scatter(log_chi, log_rmse, c="black")
ax.set_title("RMSE against fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\log(\Delta \chi$) (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$\log(\text{RMSE}_h)$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.legend()
plt.savefig(folder + "log_rmse_vs_dchi.png")
plt.close()


print("Complete.")