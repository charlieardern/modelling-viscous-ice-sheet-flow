import numpy as np
from solvers.solvers import numerical_fv_implicit_general_b
import os

N = 100
x_G_0 = 0.3
alpha = 0.24
nu = 80
num_steps = 100
num_microsteps = 10
t_final = 200
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150

# initial state:
h_0 = 1-np.linspace(0,1,num=N)+0.002+0.1

print("Computing solution...")
#x, h, x_shelf, H_shelf = numerical_fv_implicit(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, a, L)

x = np.linspace(0,20, num=5000)
b = 1.666104322801496*x

x, h, x_shelf, H_shelf = numerical_fv_implicit_general_b(h_0, x, b, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, a, L)

folder = "saved_objects/"
os.makedirs(folder, exist_ok=True)

np.save(folder + "full_h.npy", h)
np.save(folder + "full_x.npy", x)
np.save(folder + "full_H_shelf.npy", H_shelf)
np.save(folder + "full_x_shelf.npy", x_shelf)
print("Complete.")