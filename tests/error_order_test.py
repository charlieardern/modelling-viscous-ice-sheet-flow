import numpy as np
from solvers.solvers import numerical_fv_implicit
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

#N = 100
x_G_0 = 0.3
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
alpha = 0.5*np.sqrt(g_prime/g) # alpha required for A=1
print(alpha)

A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

print("Computing error order tests...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

# Run test for delta_chi order -------------------------------------------

# N_list = np.round(np.linspace(40, 100, num=5)).astype(int)
# rmse_list = []
# num_microsteps = 12000

# for i in tqdm(range(len(N_list))):
#     N = N_list[i]
#     h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
#     x, h, _, _ = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, sheet_accumulation=False, hide_output=True)
#     # Non-dimensionalise x and h:
#     x_pred = x[:,-1]/(2*B*C**4)
#     h_pred = h[:,-1]/(B*C)
#     h_true = 1 - x_pred
#     rmse_list.append(np.sqrt(np.mean((h_pred-h_true)**2)))
#     print(f"h.shape: {h_pred.shape}")
#     print(f"h.shape: {h_true.shape}")
#     print(f"Error with N = {N}: {rmse_list[i]}")

# delta_chi_values = 1/np.array(N_list)
# plt.figure(figsize=(8,5), dpi=300)
# plt.plot(delta_chi_values, rmse_list)
# plt.title("RMSE for steady state", fontname="Latin Modern Roman", fontsize=16)
# plt.xlabel(r"$\Delta \hat \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
# plt.ylabel(r"$RMSE_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
# plt.savefig(folder + "rmse_vs_dchi.png")
# plt.close()

# Run test for delta_t order -------------------------------------------

#microstep_list = np.round(np.linspace(8000, 20000, num=5)).astype(int)
microstep_list = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
rmse_list = []
N = 200
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
# _, h_fine, _, _ = numerical_fv_implicit(h_0, 10000, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
# h_fine = h_fine[:,-1]/(B*C)

for i in tqdm(range(len(microstep_list))):
    num_microsteps = microstep_list[i]
    x, h, _, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True, sheet_accumulation=False)
    
    h_fine = 1 - x[:,-1]/(2*B*C**4)
    
    # Non-dimensionalise h:
    h_pred = h[:,-1]/(B*C)
    rmse_list.append(np.sqrt(np.mean((h_pred-h_fine)**2)))
    print(f"rmse: {rmse_list[i]}")

dt_scaled_values = (a*L/(2*B**2*C**5))*t_final/(num_steps*np.array(microstep_list))
plt.figure(figsize=(8,5), dpi=300)
plt.plot(dt_scaled_values, rmse_list)
plt.title("RMSE for steady state", fontname="Latin Modern Roman", fontsize=16)
plt.xlabel(r"$\Delta \hat t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.ylabel(r"$RMSE_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
plt.savefig(folder + "rmse_vs_dt.png")
plt.close()

print("Complete.")