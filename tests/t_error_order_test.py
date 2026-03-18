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

compute_fine_time = True
compute_rmse = True

x_G_0 = 0.5
nu=800
num_steps = 100
t_final = 800
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
alpha = 0.07#0.5*np.sqrt(g_prime/g) # alpha required for A=1
print(alpha)

A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho
D = 2*B*C**4/L

print(f"A: {A}")
print(f"D: {D}")

print("Computing error order tests...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

# Run test for delta_t order -------------------------------------------
microstep_list = [10, 12, 14, 20, 30, 40, 50, 65, 80, 100]
fine_microsteps = 2000
print(f"Fine grained solution timestep: {(a*L/(2*B**2*C**5))*t_final/(num_steps*fine_microsteps)}")
rmse_list = []
N = 1500
#h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N))

if compute_fine_time:
    x_fine, h_fine, _, _ = numerical_fv_implicit(h_0, fine_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
    h_fine = h_fine[:,-1]/(B*C)
    x_fine = x_fine[:,-1]
    os.makedirs("saved_objects/", exist_ok=True)
    np.save("saved_objects/fine_solution_t.npy", h_fine)
    np.save("saved_objects/fine_solution_tx.npy", x_fine)
else:
    h_fine = np.load("saved_objects/fine_solution_t.npy")
    x_fine = np.load("saved_objects/fine_solution_tx.npy")

if compute_rmse:
    for i in tqdm(range(len(microstep_list))):
        num_microsteps = microstep_list[i]
        x, h, _, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
        x_pred = x[:,-1]
        # Non-dimensionalise h:
        h_pred = h[:,-1]/(B*C)
        h_fine_i = np.interp(x_pred, x_fine, h_fine)
        
        rmse_list.append(np.sqrt(np.mean((h_pred-h_fine_i)**2)))
        print(f"rmse: {rmse_list[i]}")
    np.save("saved_objects/rmse_t.npy", np.array(rmse_list))
else:
    rmse_list = np.load("saved_objects/rmse_t.npy")

dt_scaled_values = (a*L/(2*B**2*C**5))*t_final/(num_steps*np.array(microstep_list))

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)

ax.scatter(dt_scaled_values, rmse_list, c="black")

popt, pcov = curve_fit(linear, dt_scaled_values, rmse_list)
ax.plot(dt_scaled_values, linear(dt_scaled_values,*popt), c="violet", label="Linear fit")

ax.set_title("RMSE against fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$\text{RMSE}_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.legend()
plt.savefig(folder + "rmse_vs_dt.png")
plt.close()

# Log plots

log_t = np.log(dt_scaled_values)
log_rmse = np.log(rmse_list)

popt, pcov = curve_fit(linear, log_t, log_rmse)

fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4), dpi=300)

ax.scatter(log_t, log_rmse, c="black")
ax.plot(log_t, linear(log_t, *popt), label=f"Linear fit with \nm = {popt[0]:.2f}", c="violet")
ax.set_title("RMSE against fine-grained solution - log plot", fontname="Latin Modern Roman", fontsize=16)
ax.set_xlabel(r"$\log\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$\log\text{RMSE}_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.legend()
plt.savefig(folder + "log_rmse_vs_dt.png")
plt.close()