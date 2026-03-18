import numpy as np
from solvers.solvers import numerical_fv_implicit
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
alpha = 0.24
nu = 160
num_steps = 100
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
max_dt = 0.003
min_dt = 0.0005

max_dchi = 0.03
min_dchi = 0.005

timesteps = np.linspace(max_dt, min_dt, num=200)
chisteps = np.linspace(min_dchi, max_dchi, num=200)

t_scaler = 900
scaled_t_steps = t_scaler*(a*L/(2*B**2*C**5))*timesteps

print(f"true t max: {t_scaler*(a*L/(2*B**2*C**5))*max_dt}")
print(f"true t min: {t_scaler*(a*L/(2*B**2*C**5))*min_dt}")

print("Computing stability plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

chi_range = max(chisteps)-min(chisteps)
t_range = max(scaled_t_steps)-min(scaled_t_steps)

boundary_cross = False

boundary_dt = []
boundary_dchi = []

fig, ax = plt.subplots(constrained_layout=True, figsize=(4.5,4.5), dpi=400)
ax.set_xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
ax.set_ylabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize = 14)
ax.set_title("Solver stability boundary", fontname="Latin Modern Roman", fontsize = 16)
ax.set_xlim(min(chisteps)-0.1*chi_range, max(chisteps)+0.1*chi_range)
ax.set_ylim(min(scaled_t_steps)-0.1*t_range, max(scaled_t_steps)+0.1*t_range)
ax.text(0.008, 0.01, "Unstable", fontname = "Latin Modern Roman", fontsize = 14)
ax.text(0.022, 0.004, "Stable", fontname = "Latin Modern Roman", fontsize = 14)

print("Calculating reference solution...")
N_true = 100
microsteps_true = 5000

if compute_new:
    #h_0 = 1-np.linspace(0,1,num=N_true)+0.002+0.1
    h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N_true))
    true_x, true_h, converge, _ = numerical_fv_implicit(h_0, microsteps_true, num_steps, t_scaler*t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
    print(f"Reference solution converged: {converge}")
    for chi_idx in tqdm(range(len(chisteps))):
        boundary_crossed = False
        
        for t_idx in range(len(timesteps)):
            delta_t = timesteps[t_idx]
            delta_chi = chisteps[chi_idx]
            #h_0 = 1-np.linspace(0,1,num=round(1/delta_chi))+0.002+0.1
            h_0 = 0.7*(1-0.9*np.linspace(0,1,num=round(1/delta_chi)))
            num_microsteps = round(t_final/(num_steps*delta_t))
            if not boundary_crossed:
                # Wrapper hides all divergence output from solver
                with np.errstate(all="ignore"):
                    x, h, converge, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_scaler*t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True, hide_output=True)
                    err = np.sqrt(np.mean((h[:,-1]-np.interp(x[:,-1], true_x[:,-1], true_h[:,-1]))**2))
                    #print(f"delta_t = {delta_t} | delta_chi = {delta_chi} | converge = {converge}")
                if converge and err < 0.01:
                    print(f"Converged with error {err}")
                    boundary_dt.append(scaled_t_steps[t_idx])
                    boundary_dchi.append(1/np.round(1/delta_chi))
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