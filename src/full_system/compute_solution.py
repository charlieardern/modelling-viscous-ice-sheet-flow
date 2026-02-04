import numpy as np
import matplotlib.pyplot as plt
from solvers import numerical_fv

N = 1000
x_G_0 = 1.0
alpha = 0.17
nu = 700
q_0 = 1
num_steps = 100
num_microsteps = 10000
t_final = 200
rho_w = 1000
rho = 917
g = 9.81

# initial state:
#h_0 = np.sqrt(1-np.linspace(0,1,num=N))+0.002
h_0 = 1-np.linspace(0,1,num=N)+0.002
x, h, x_shelf, H_shelf = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, q_0)
print("Saving files...")

np.save("saved_objects/full_h.npy", h)
np.save("saved_objects/full_x.npy", x)
np.save("saved_objects/full_H_shelf.npy", H_shelf)
np.save("saved_objects/full_x_shelf.npy", x_shelf)