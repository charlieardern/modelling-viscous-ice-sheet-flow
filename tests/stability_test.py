import numpy as np
from solvers.solvers import numerical_fv_implicit
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

#N = 100
x_G_0 = 0.3
alpha = 0.24
nu = 80
num_steps = 100
#num_microsteps = 12000
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

# New
max_dt = 0.01
min_dt = 0.001

max_dchi = 0.1
min_dchi = 0.01

# timesteps = (1-np.linspace(0,1,num=300)**2)*max_dt + min_dt
# chisteps = (np.linspace(0,1,num=300)**2)*max_dchi + min_dchi

timesteps = np.linspace(max_dt, min_dt, num=500)
chisteps = np.linspace(min_dchi, max_dchi, num=500)

t_scaler = 900
scaled_t_steps = t_scaler*(a*L/(2*B**2*C**5))*timesteps

print("Computing stability plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

chi_range = max(chisteps)-min(chisteps)
t_range = max(scaled_t_steps)-min(scaled_t_steps)

boundary_cross = False

boundary_dt = []
boundary_dchi = []

plt.figure(figsize=(6,6), dpi=400)
plt.xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
plt.ylabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
plt.title("Solver stability", fontname="Latin Modern Roman", fontsize = 16)
plt.xlim(min(chisteps)-0.1*chi_range, max(chisteps)+0.1*chi_range)
plt.ylim(min(scaled_t_steps)-0.1*t_range, max(scaled_t_steps)+0.1*t_range)
plt.text(0.02, 0.05, "Unstable", fontname = "Latin Modern Roman", fontsize = 14)
plt.text(0.08, 0.02, "Stable", fontname = "Latin Modern Roman", fontsize = 14)

for t_idx in tqdm(range(len(timesteps))):
    boundary_crossed = False
    for chi_idx in range(len(chisteps)):
        delta_t = timesteps[t_idx]
        delta_chi = chisteps[chi_idx]
        h_0 = 1-np.linspace(0,1,num=round(1/delta_chi))+0.002+0.1
        num_microsteps = round(t_final/(num_steps*delta_t))
        if not boundary_crossed:
            # Wrapper hides all divergence output from solver
            with np.errstate(all="ignore"):
                _, _, converge, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_scaler*t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True, hide_output=True)
            if converge:
                boundary_dt.append(scaled_t_steps[t_idx])
                boundary_dchi.append(delta_chi)
                boundary_crossed = True
    if len(boundary_dchi) > 1:
        plt.plot([boundary_dchi[-1], boundary_dchi[-2]], [boundary_dt[-1], boundary_dt[-2]], c="black")
        plt.savefig(folder + "stability_plot_hr_sweep.png")
plt.close()

boundary_dt = []
boundary_dchi = []

plt.figure(figsize=(6,6), dpi=400)
plt.xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
plt.ylabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
plt.title("Solver stability", fontname="Latin Modern Roman", fontsize = 16)
plt.xlim(min(chisteps)-0.1*chi_range, max(chisteps)+0.1*chi_range)
plt.ylim(min(scaled_t_steps)-0.1*t_range, max(scaled_t_steps)+0.1*t_range)
plt.text(0.02, 0.05, "Unstable", fontname = "Latin Modern Roman", fontsize = 14)
plt.text(0.08, 0.02, "Stable", fontname = "Latin Modern Roman", fontsize = 14)

timesteps = np.linspace(max_dt, min_dt, num=100)
chisteps = np.linspace(min_dchi, max_dchi, num=100)

for chi_idx in tqdm(range(len(chisteps))):
    boundary_crossed = False
    for t_idx in range(len(timesteps)):
        delta_t = timesteps[t_idx]
        delta_chi = chisteps[chi_idx]
        h_0 = 1-np.linspace(0,1,num=round(1/delta_chi))+0.002+0.1
        num_microsteps = round(t_final/(num_steps*delta_t))
        if not boundary_crossed:
            # Wrapper hides all divergence output from solver
            with np.errstate(all="ignore"):
                _, _, converge, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_scaler*t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True, hide_output=True)
            if converge:
                boundary_dt.append(scaled_t_steps[t_idx])
                boundary_dchi.append(delta_chi)
                boundary_crossed = True
    if len(boundary_dchi) > 1:
        plt.plot([boundary_dchi[-1], boundary_dchi[-2]], [boundary_dt[-1], boundary_dt[-2]], c="black")
        plt.savefig(folder + "stability_plot_vr_sweep.png")
plt.close()

print("Complete.")