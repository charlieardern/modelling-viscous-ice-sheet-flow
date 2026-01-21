import numpy as np
from analytical import compute_analytical
from numerical import compute_numerical_explicit, compute_numerical_implicit

def h_0(x):
    return -h_G*x/L + 2*h_G

pi = np.pi
h_G = 2.5
L = 4
a = 0.01
k = 0.02
num_fourier_terms = 20
num_t_steps = 100
num_microsteps = 20 # Number of steps to compute per time step (true time step=num_t_steps*num_microsteps)
t_final = 1000
dim = 100
w = L/dim

x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1, 1)
t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1, 1)

h_an = compute_analytical(num_fourier_terms, num_t_steps, t_final, dim, L, h_G, a, k)
h_num_ex = compute_numerical_explicit(h_0, num_microsteps, num_t_steps, t_final, dim, L, h_G, a, k)
h_num_im = compute_numerical_implicit(h_0, num_microsteps, num_t_steps, t_final, dim, L, h_G, a, k)

np.save("saved_objects/linear_t.npy", t)
np.save("saved_objects/linear_x.npy", x)
np.save("saved_objects/linear_analytical_solution.npy", h_an)
np.save("saved_objects/linear_numerical_solution_ex.npy", h_num_ex)
np.save("saved_objects/linear_numerical_solution_im.npy", h_num_im)