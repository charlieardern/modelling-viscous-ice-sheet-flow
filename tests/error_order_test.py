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
    return a*x**2+b*x + c

time_test = True
compute_fine_time = False

space_test = True
compute_fine_space = False

x_G_0 = 0.3
#nu = 80
nu=300
num_steps = 100
#num_microsteps = 12000
t_final = 100
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
alpha = 0.24#0.5*np.sqrt(g_prime/g) # alpha required for A=1
print(alpha)

A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho

print("Computing error order tests...")

folder = "figures/"
os.makedirs(folder, exist_ok=True)

if time_test:
    # Run test for delta_t order -------------------------------------------
    microstep_list = [14, 15, 16, 17, 20, 40, 80, 90, 100]
    fine_microsteps = 1000
    rmse_list = []
    N = 1000
    h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
    # _, h_fine, _, _ = numerical_fv_implicit(h_0, 10000, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
    # h_fine = h_fine[:,-1]/(B*C)

    if compute_fine_time:
        h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1
        _, h_fine, _, _ = numerical_fv_implicit(h_0, fine_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)
        h_fine = h_fine[:,-1]/(B*C)
        os.makedirs("saved_objects/", exist_ok=True)
        np.save("saved_objects/fine_solution_t.npy", h_fine)
    else:
        h_fine = np.load("saved_objects/fine_solution_t.npy")

    for i in tqdm(range(len(microstep_list))):
        num_microsteps = microstep_list[i]
        x, h, _, _ = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L, test_mode=True)

        
        # Non-dimensionalise h:
        h_pred = h[:,-1]/(B*C)
        rmse_list.append(np.sqrt(np.mean((h_pred-h_fine)**2)))
        print(f"rmse: {rmse_list[i]}")

    dt_scaled_values = (a*L/(2*B**2*C**5))*t_final/(num_steps*np.array(microstep_list))
    plt.figure(figsize=(8,5), dpi=300)
    plt.scatter(dt_scaled_values, rmse_list)

    popt, pcov = curve_fit(quadratic, dt_scaled_values, rmse_list)
    plt.plot(dt_scaled_values, quadratic(dt_scaled_values,*popt))

    #plt.plot(np.log(dt_scaled_values), np.log(rmse_list))
    plt.title("RMSE for steady state", fontname="Latin Modern Roman", fontsize=16)
    plt.xlabel(r"$\Delta \hat t$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    plt.ylabel(r"$RMSE_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    plt.savefig(folder + "rmse_vs_dt.png")
    plt.close()

    
    print(f"Temporal quadratic weight: {popt[0]} +- {np.sqrt(pcov[0,0])}")
    print(f"Temporal linear weight: {popt[1]} +- {np.sqrt(pcov[1,1])}")

# Run test for delta_chi order -------------------------------------------
if space_test:
    t_final = 100
    num_microsteps = 1000
    N_values = np.round(1/np.linspace(1/5, 1/100, num=30))
    #N_values = np.round(np.linspace(5,100, num=30))
    N_fine = 1000
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

    plt.figure(figsize=(8,5), dpi=300)
    plt.plot(1/np.array(N_values), rmse_list)
    plt.title("RMSE for steady state", fontname="Latin Modern Roman", fontsize=16)
    plt.xlabel(r"$\Delta \hat \chi$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    plt.ylabel(r"$RMSE_h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    plt.savefig(folder + "rmse_vs_dchi.png")
    plt.close()
    popt, pcov = curve_fit(quadratic, 1/np.array(N_values), rmse_list)
    print(f"Spatial quadratic weight: {popt[0]} +- {np.sqrt(pcov[0,0])}")
    print(f"Spatial linear weight: {popt[1]} +- {np.sqrt(pcov[1,1])}")

print("Complete.")