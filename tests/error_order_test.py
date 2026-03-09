import numpy as np
from solvers.solvers import numerical_fv
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

#N = 100
x_G_0 = 0.3
nu = 80
num_steps = 100
#num_microsteps = 12000
t_final = 400
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
alpha = 0.5*np.sqrt(g_prime/g) # alpha required for A=1
print(alpha)

A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

print("Computing steady state plot...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

# Run test for delta_chi order -------------------------------------------

N_list = np.round(np.linspace(20, 100, num=20)).astype(int)
rmse_list = []
num_microsteps = 12000

for i in tqdm(range(len(N_list))):
    N = N_list[i]
    h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
    x, h, _, _ = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, sheet_accumulation=False)

    # Non-dimensionalise x and h:
    x_pred = x[:,-1]/(2*B*C**4)
    h_pred = h[:,-1]/(B*C)
    h_true = 1 - x_pred
    rmse_list.append(np.sqrt(np.mean((h_pred-h_true)**2)))

delta_chi_values = 1/np.array(N_list)
plt.figure(figsize=(8,5), dpi=300)
plt.plot(delta_chi_values, rmse_list)
plt.title("RMSE for steady state", fontname="Latin Modern Roman", fontsize=16)
plt.xlabel(r"$\Delta \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$RMSE_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.savefig(folder + "rmse_vs_dchi.png")
plt.close()

# Run test for delta_t order -------------------------------------------

microstep_list = np.round(np.linspace(10000, 14000, num=20)).astype(int)
rmse_list = []
N = 100
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

for i in tqdm(range(len(microstep_list))):
    num_microsteps = microstep_list[i]
    x, h, _, _ = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, sheet_accumulation=False)

    # Non-dimensionalise x and h:
    x_pred = x[:,-1]/(2*B*C**4)
    h_pred = h[:,-1]/(B*C)
    h_true = 1 - x_pred
    rmse_list.append(np.sqrt(np.mean((h_pred-h_true)**2)))

dt_scaled_values = (a*L/(2*B**2*C**5))*t_final/(num_steps+np.array(microstep_list))
plt.figure(figsize=(8,5), dpi=300)
plt.plot(dt_scaled_values, rmse_list)
plt.title("RMSE for steady state", fontname="Latin Modern Roman", fontsize=16)
plt.xlabel(r"$\Delta \t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$RMSE_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.savefig(folder + "rmse_vs_dt.png")
plt.close()

print("Complete.")