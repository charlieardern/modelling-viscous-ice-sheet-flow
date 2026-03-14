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

x_G_0 = 0.3
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

print("Computing error order tests...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

# Run test for delta_t order -------------------------------------------
microstep_list = [10, 14, 20, 40, 50, 100, 500, 1000]
fine_microsteps = 2000
print(f"Fine grained solution timestep: {(a*L/(2*B**2*C**5))*t_final/(num_steps*fine_microsteps)}")
rmse_list = []
N = 1500
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

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
    np.save("saved_objects/remse_t.npy", np.array(rmse_list))
else:
    rmse_list = np.load("saved_objects/rmse_t.npy")

dt_scaled_values = (a*L/(2*B**2*C**5))*t_final/(num_steps*np.array(microstep_list))
plt.figure(figsize=(8,5), dpi=300)
plt.scatter(dt_scaled_values, rmse_list)

popt, pcov = curve_fit(quadratic, dt_scaled_values, rmse_list)
plt.plot(dt_scaled_values, quadratic(dt_scaled_values,*popt))

#plt.plot(np.log(dt_scaled_values), np.log(rmse_list))
plt.title("RMSE to fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
plt.xlabel(r"$\Delta t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$\text{RMSE}_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.savefig(folder + "rmse_vs_dt.png")
plt.close()

print(f"Temporal quadratic weight: {popt[0]} +- {np.sqrt(pcov[0,0])}")
print(f"Temporal linear weight: {popt[1]} +- {np.sqrt(pcov[1,1])}")

# Log plots

log_t = np.log(dt_scaled_values)
log_rmse = np.log(rmse_list)

print(f"log(t): {log_t}")
print(f"log(rmse): {log_rmse}")

popt, pcov = curve_fit(linear, log_t, log_rmse)

plt.figure(figsize=(8,5), dpi=300)

plt.scatter(log_t, log_rmse)
plt.plot(log_t, linear(log_t, *popt), label=f"m = {popt[0]}")
plt.title("RMSE to fine-grained solution", fontname="Latin Modern Roman", fontsize=16)
plt.xlabel(r"$\log(\Delta t)$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$\log(\text{RMSE}_h)$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.legend()
plt.savefig(folder + "log_rmse_vs_dt.png")
plt.close()