import numpy as np
from solvers.solvers import IceSheetSolver
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.optimize import curve_fit

def linear(x, a, b):
    return a*x + b

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

compute_new = True

x_G_0 = 0.3
nu = 160
L = 150
D = 0.5
t_final = 300
A = 1.5

max_dt = 0.0175
min_dt = 0.002

max_dchi = 0.03
min_dchi = 0.005

# Set timesteps, chisteps
timesteps = np.linspace(max_dt, min_dt, num=200)
chisteps = np.linspace(min_dchi, max_dchi, num=200)

# Round to respective integers
microsteps_list = np.round(t_final/(100*timesteps)).astype(int)
N_list = np.round(1/chisteps).astype(int)

# Reassign to timesteps, chisteps
timesteps = t_final/(100*microsteps_list)
chisteps = 1/N_list

print("Computing stability plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

chi_range = max(chisteps)-min(chisteps)
t_range = max(timesteps)-min(timesteps)

boundary_cross = False

boundary_dt = []
boundary_dchi = []

fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5,4.5), dpi=400)
ax.set_xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
ax.set_ylabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
ax.set_title("Solver stability boundary", fontname="Latin Modern Roman", fontsize = 16)
ax.set_xlim(min(chisteps)-0.1*chi_range, max(chisteps)+0.1*chi_range)
ax.set_ylim(min(timesteps)-0.1*t_range, max(timesteps)+0.1*t_range)
ax.text(0.008, 0.01, "Unstable", fontname = "Latin Modern Roman", fontsize = 14)
ax.text(0.022, 0.004, "Stable", fontname = "Latin Modern Roman", fontsize = 14)

print("Calculating reference solution...")
N_true = 100
microsteps_true = 5000

x_bed = np.linspace(0,20, num=2000)
b_bed = A*x_bed

if compute_new:
    h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N_true))
    true_system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, microsteps_true, t_final)
    true_system.compute_solution(compute_shelf=False)
    true_h = true_system.h_tnsr[:,-1]
    plt.figure()
    plt.plot(true_system.chi, true_h)
    plt.savefig("test.png")
    plt.close()
    print(f"Reference solution converged: {true_system.converge}")
    for chi_idx in tqdm(range(len(chisteps))):
        boundary_crossed = False
        
        for t_idx in range(len(timesteps)):
            print(f"chi_idx: {chi_idx} | t_idx: {t_idx}")
            N_i = N_list[chi_idx]
            microsteps_i = microsteps_list[t_idx]
            delta_t = timesteps[t_idx]
            delta_chi = chisteps[chi_idx]
            h_0 = 0.7*(1-0.9*np.linspace(0,1,N_i))

            if not boundary_crossed:
                # Wrapper hides all divergence output from solver
                h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N_i))
                system_i = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, microsteps_i, t_final)
                with np.errstate(all="ignore"):
                    system_i.compute_solution(compute_shelf=False)
                h_i = system_i.h_tnsr[:,-1]
                converge_i = system_i.converge
                err = np.sqrt(np.mean((h_i-np.interp(system_i.chi, true_system.chi, true_h))**2))
                print(f"err: {err} with N = {N_i} and microsteps = {microsteps_i}")
                plt.figure()
                plt.plot(system_i.chi, h_i)
                plt.savefig("test.png")
                plt.close()
                if converge_i and err < 0.01:
                    print(f"Converged with error {err}")
                    boundary_dt.append(delta_t)
                    boundary_dchi.append(1/N_i)
                    boundary_crossed = True
        if len(boundary_dchi) > 1:
            ax.plot([boundary_dchi[-1], boundary_dchi[-2]], [boundary_dt[-1], boundary_dt[-2]], c="blue")
            plt.savefig(folder + "stability_plot.png")
    np.save("saved_objects/dt_stability.npy", np.array(boundary_dt))
    np.save("saved_objects/dchi_stability.npy", np.array(boundary_dchi))
else:
    boundary_dt = np.load("saved_objects/dt_stability.npy")
    boundary_dchi = np.load("saved_objects/dchi_stability.npy")
    ax.scatter(boundary_dchi, boundary_dt, c="black", s=2)
    popt, pcov = curve_fit(linear, boundary_dchi, boundary_dt)
    ax.plot(boundary_dchi, linear(boundary_dchi, *popt), c='orange', label="Linear fit")
    plt.legend()
    plt.savefig(folder + "stability_plot.png")
    plt.close()

# Log plot

log_dchi = np.log(boundary_dchi)
log_dt = np.log(boundary_dt)

fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5,4.5), dpi=400)
ax.set_xlabel(r"$\log\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
ax.set_ylabel(r"$\log\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
ax.text(-5, -5, "Unstable", fontname = "Latin Modern Roman", fontsize = 14)
ax.text(-4.25, -5.75, "Stable", fontname = "Latin Modern Roman", fontsize = 14)
ax.set_title("Solver stability boundary - log plot", fontname="Latin Modern Roman", fontsize = 16)
ax.scatter(log_dchi, log_dt, s=2, c="black")
popt, pcov = curve_fit(linear, log_dchi, log_dt)
ax.plot(log_dchi, linear(log_dchi, *popt), label=f"Linear fit \nm={popt[0]:.2f}", c="orange")
plt.legend()
plt.savefig(folder + "log_stability_plot.png")


print("Complete.")