import numpy as np

from solvers import compute_numerical_explicit

def s_0(x, s_G):
    return -s_G*x/L + 2*s_G

pi = np.pi
s_G = 2.5
L = 4
a = 0.01
num_t_steps = 100
num_microsteps = 1000 # Number of steps to compute per time step (true time step=num_t_steps*num_microsteps)
t_final = 10
dim = 100
w = L/dim
g = 9.81
eta = 20

bed = np.zeros(dim).reshape(dim,1)

x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1, 1)
t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1, 1)

h_num = compute_numerical_explicit(s_0, bed, num_microsteps, num_t_steps, t_final, dim, L, s_G, a, g, eta)

np.save("saved_objects/nonlinear_t.npy", t)
np.save("saved_objects/nonlinear_x.npy", x)
np.save("saved_objects/numerical_solution.npy", h_num)