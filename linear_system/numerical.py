import numpy as np

print("Running numerical.py...")
h_G = 2.5
L = 4
a = 0.02
k = 0.01
w = 1

dim = 100
num_t_steps = 400

pi = np.pi

x = np.linspace(0,L, num=dim).reshape(-1, 1)
t = np.linspace(0,99, num=num_t_steps).reshape(1,-1)

delta_t = t[0,1]-t[0,0]
print(delta_t)

A = np.eye(dim, k=1)+np.eye(dim, k=-1)-2*np.eye(dim)
A[0,0] = -1
A[-1,-1] = -3
A = A*k/w**2

B = a*np.ones(dim)
B[-1]+= 2*h_G*k/w**2
print(B)
B = B.reshape(-1,1)

def h_0(x):
    return -h_G*x/L + 2*h_G

h_list = []
h_i = h_0(x)
h_list.append(h_i)

for i in range(num_t_steps-1):
    h_i = h_i + (A@h_i+B)*delta_t
    h_list.append(h_i)

h = np.concatenate(h_list, axis=1)

np.save("saved_objects/linear_numerical_solution.npy", h)