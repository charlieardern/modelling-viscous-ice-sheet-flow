import numpy as np
import matplotlib.pyplot as plt

print("Running analytical.py...")
h_G = 2.5
L = 4
a = 0.01
k = 0.01
num_terms = 10
num_t_steps=100
t_final=200
dim=100
w=L/dim

pi = np.pi

x = np.linspace(0+0.5*w,L-0.5*w, num=dim).reshape(-1, 1, 1)
t = np.linspace(0,t_final, num=num_t_steps).reshape(1,-1, 1)
n = np.arange(0,num_terms, 1).reshape(1,1,-1)

def A(n):
    f_1 = 8*h_G/(pi**2*(2*n+1)**2)
    f_2 = -16*a*L**2*(-1)**n/(k*pi**3*(2*n+1)**3)
    return f_1 + f_2

A_n = A(n)

# Compute fourier terms and sum them up:
h_3_tnsr = np.exp(-pi**2*(2*n+1)**2*k*t/(4*L**2))*A_n*np.cos(pi*x*(2*n+1)/(2*L))
h_f_terms = h_3_tnsr.sum(axis=2).reshape(h_3_tnsr.shape[0], h_3_tnsr.shape[1], 1)

# Add to rest of terms to give final solution:
h = h_f_terms - a*x**2/(2*k)+h_G+a*L**2/(2*k)

def h_0(x):
    return -h_G*x/L + 2*h_G

plt.plot(x[:,0,0], h_0(x[:,0,0]))
plt.plot(x[:,0,0], h[:,0,0])
plt.savefig("figures/test1")

np.save("saved_objects/linear_analytical_t.npy", t)
np.save("saved_objects/linear_analytical_x.npy", x)
np.save("saved_objects/linear_analytical_solution.npy", h)
print("Complete.")