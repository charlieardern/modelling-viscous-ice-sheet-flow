import numpy as np
from solvers.solvers import IceSheetSolver
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

print("Computing chi error order test...")

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def linear(x, a, b):
    return a*x + b

compute_fine_space = True
compute_rmse = True

# System setup --------------------------------------------

x_G_0 = 0.5
nu = 80
t_final = 0.8
num_microsteps = 3000
L = 150
A = 2
D = 0.4

x_bed = np.linspace(0,20, num=2000)
b_bed = A*x_bed

# Compute fine space --------------------------------------

N_fine = 700
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N_fine))
fine_system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)

if compute_fine_space:
    fine_system.compute_solution(compute_shelf=False)
    h_fine = fine_system.h_tnsr[:,-1]
    x_G_fine = fine_system.x_G_tnsr[:,-1]
    os.makedirs("saved_objects/", exist_ok=True)
    np.save("saved_objects/fine_solution_chi.npy", h_fine)
    np.save("saved_objects/fine_solution_chi_x_G.npy", x_G_fine)
else:
    h_fine = np.load("saved_objects/fine_solution_chi.npy")
    x_G_fine = np.load("saved_objects/fine_solution_chi_x_G.npy")

# Run tests -------------------------------------------

N_values = np.array([3, 4, 5, 6, 7, 8, 10, 20, 40, 80])
rmse_list = []
x_G_rmse_list = []

if compute_rmse:
    for i in tqdm(range(len(N_values))):
        h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N_values[i]))
        system_i = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)
        system_i.compute_solution(compute_shelf=False)

        h_i = system_i.h_tnsr[:,-1]
        h_fine_i = np.interp(system_i.chi, fine_system.chi, h_fine)
        rmse_list.append(np.sqrt(np.mean((h_i-h_fine_i)**2)))

        x_G_i = system_i.x_G_tnsr[:,-1]
        x_G_rmse_list.append((np.sqrt(x_G_i-x_G_fine)**2))
        print(f"rmse: {rmse_list[i]} | x_G_rmse: {x_G_rmse_list[i]}")
        np.save("saved_objects/rmse_chi.npy", rmse_list)
        np.save("saved_objects/rmse_chi_x_G.npy", x_G_rmse_list)
else:
    rmse_list = np.load("saved_objects/rmse_chi.npy")
    x_G_rmse_list = np.load("saved_objects/rmse_chi_x_G.npy")

# Plot results --------------------------------------------

os.makedirs("figures/", exist_ok=True)

dchi_values = 1/N_values
popt, pcov = curve_fit(quadratic, dchi_values, rmse_list)

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)
ax.set_title("RMSE against fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"RMSE$_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)

ax.plot(np.linspace(dchi_values[-1],dchi_values[0], num=200), quadratic(np.linspace(dchi_values[-1],dchi_values[0], num=200), *popt), c="red", label="Quadratic fit")
ax.scatter(dchi_values, rmse_list, c="black")
ax.legend()

plt.savefig("figures/" + "rmse_vs_dchi.png")
plt.close()

# Log plots -----------------------------------------------

log_chi = np.log(dchi_values)
log_rmse = np.log(rmse_list)
popt, pcov = curve_fit(linear, log_chi, log_rmse)

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)
ax.set_title("RMSE against fine-grained solution - log plot", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\log\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$\log\text{RMSE}_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)

ax.plot(log_chi, linear(log_chi, *popt), label=f"Linear fit with \nm = {popt[0]:.2f}", c="red")
ax.scatter(log_chi, log_rmse, c="black")
plt.legend()

plt.savefig("figures/" + "log_rmse_vs_dchi.png")
plt.close()

# x_G log plots -----------------------------------------------

# log_chi = np.log(dchi_values)
# log_rmse_x_G = np.log(x_G_rmse_list)
# popt, pcov = curve_fit(linear, log_chi, log_rmse_x_G)

# fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)
# ax.set_title(r"$x_G$ RMSE against fine-grained solution - log plot", fontname="Latin Modern Roman", fontsize=16)
# ax.set_xlabel(r"$\log\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
# ax.set_ylabel(r"$\log\text{RMSE}_{x_G}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)

# ax.plot(log_chi, linear(log_chi, *popt), label=f"Linear fit with \nm = {popt[0]:.2f}", c="red")
# ax.scatter(log_chi, log_rmse_x_G, c="black")
# plt.legend()

# plt.savefig("figures/" + "log_rmse_vs_dchi_x_G.png")
# plt.close()

print("Complete.")