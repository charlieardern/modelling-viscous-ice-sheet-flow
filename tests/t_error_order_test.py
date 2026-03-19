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

print("Computing t error order test...")

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def linear(x, a, b):
    return a*x + b

compute_fine_time = False
compute_rmse = True

# System setup --------------------------------------------

x_G_0 = 0.1
nu = 80
t_final = 0.2
L = 150
A = 0.8
D = 0.4
N = 800
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N))

x_bed = np.linspace(0,20, num=2000)
b_bed = A*x_bed

# Compute fine system -------------------------------------

fine_microsteps = 2000
fine_system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, fine_microsteps, t_final)

if compute_fine_time:
    fine_system.compute_solution(compute_shelf=False)
    h_fine = fine_system.h_tnsr[:,-1]
    x_G_fine = fine_system.x_G_tnsr[:,-1]
    os.makedirs("saved_objects/", exist_ok=True)
    np.save("saved_objects/fine_solution_t.npy", h_fine)
    np.save("saved_objects/fine_solution_t_x_G.npy", x_G_fine)
else:
    h_fine = np.load("saved_objects/fine_solution_t.npy")
    x_G_fine = np.load("saved_objects/fine_solution_t_x_G.npy")

# Run tests -----------------------------------------------

microstep_list = np.array([5, 6, 7, 8, 9, 10, 12, 14, 20, 30, 40, 50, 65, 80, 100])
rmse_list = []
x_G_rmse_list = []

if compute_rmse:
    for i in tqdm(range(len(microstep_list))):
        num_microsteps = microstep_list[i]
        system_i = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)
        system_i.compute_solution(compute_shelf=False)

        h_i = system_i.h_tnsr[:,-1]
        rmse_list.append(np.sqrt(np.mean((h_i-h_fine)**2)))

        x_G_i = system_i.x_G_tnsr[:,-1]
        x_G_rmse_list.append(np.sqrt((x_G_i-x_G_fine)**2))

        print(f"rmse: {rmse_list[i]} | x_G_rmse: {x_G_rmse_list[i]}")
    np.save("saved_objects/rmse_t.npy", np.array(rmse_list))
    np.save("saved_objects/rmse_t_x_G.npy", np.array(rmse_list))
else:
    rmse_list = np.load("saved_objects/rmse_t.npy")
    x_G_rmse_list = np.load("saved_objects/rmse_t_x_G.npy")

# Plot results --------------------------------------------

dt_values = t_final/(fine_system.num_steps*microstep_list)
popt, pcov = curve_fit(linear, dt_values, rmse_list)

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)
ax.set_title("RMSE against fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$\text{RMSE}_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)

ax.scatter(dt_values, rmse_list, c="black")
ax.plot(dt_values, linear(dt_values,*popt), c="violet", label="Linear fit")
ax.legend()

plt.savefig("figures/" + "rmse_vs_dt.png")
plt.close()

# Log plots -----------------------------------------------

log_t = np.log(dt_values)
log_rmse = np.log(rmse_list)
popt, pcov = curve_fit(linear, log_t, log_rmse)


fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)
ax.set_title("RMSE against fine-grained solution - log plot", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\log\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$\log\text{RMSE}_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)


ax.scatter(log_t, log_rmse, c="black")
ax.plot(log_t, linear(log_t, *popt), label=f"Linear fit with \nm = {popt[0]:.2f}", c="violet")
ax.legend()

plt.savefig("figures/" + "log_rmse_vs_dt.png")
plt.close()

# x_G log plot ----------------------------------------------

# log_t = np.log(dt_values)
# log_rmse_x_G = np.log(x_G_rmse_list)
# popt, pcov = curve_fit(linear, log_t, log_rmse_x_G)


# fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)
# ax.set_title(r"$x_G$ RMSE against fine-grained solution - log plot", fontname="Latin Modern Roman", fontsize=16)
# ax.set_xlabel(r"$\log\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
# ax.set_ylabel(r"$\log\text{RMSE}_{x_G}$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)


# ax.scatter(log_t, log_rmse_x_G, c="black")
# ax.plot(log_t, linear(log_t, *popt), label=f"Linear fit with \nm = {popt[0]:.2f}", c="violet")
# ax.legend()

plt.savefig("figures/" + "log_rmse_x_G_vs_dt.png")
plt.close()