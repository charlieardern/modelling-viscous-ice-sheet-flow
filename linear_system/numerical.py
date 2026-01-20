import numpy as np

# We solve on a grid (0+0.5w,0+1.5w, ... , L-0.5w)

print("Running numerical.py...")
h_G = 2.5
L = 4
a = 0.01
k = 0.01

dim = 100
num_t_steps = 100
num_microsteps = 50 # Number of steps to compute per time step (true time step=num_t_steps*num_microsteps)
t_final = 200

w = L/dim

pi = np.pi

x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1)
t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1)

delta_t = (t[0,1]-t[0,0])/num_microsteps
print(f"w^2/2k: {w**2/(2*k)}")
print(f"delta_t: {delta_t}")
converge = True if delta_t<=(w**2/(2*k)) else False
print(f"Expected to converge: {converge}")

A = np.eye(dim, k=1)+np.eye(dim, k=-1)-2*np.eye(dim)
A[0,0] = -1
A[-1,-1] = -3
A = A*k/w**2

B = a*np.ones(dim)
B[-1]+= 2*h_G*k/w**2
B = B.reshape(-1,1)

def h_0(x):
    return -h_G*x/L + 2*h_G

h_list = []
h_i = h_0(x)
h_list.append(h_i)

for i in range(num_t_steps):
    for ii in range(num_microsteps):
        h_i = h_i + (A@h_i+B)*delta_t
    h_list.append(h_i)

h = np.concatenate(h_list, axis=1)

np.save("saved_objects/linear_numerical_solution.npy", h)
print("Complete.")