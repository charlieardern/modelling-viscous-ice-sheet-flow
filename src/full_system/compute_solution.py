import numpy as np
import matplotlib.pyplot as plt
from solvers import numerical_fv

N = 1000
x_G_0 = 1
alpha = 0.1
nu = 400
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
x, h = numerical_fv(h_0, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, alpha, q_0)

np.save("saved_objects/full_h.npy", h)
np.save("saved_objects/full_x.npy", x)

t = np.linspace(0, t_final, num=num_steps+1)
t_2d = np.tile(2, (N, 1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, t_2d, h, cmap='viridis')

ax.set_xlabel("x coordinate")
ax.set_ylabel("time")
ax.set_zlabel("height")

plt.savefig("figures/full_system.png")