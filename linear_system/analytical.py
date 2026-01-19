import numpy as np
import matplotlib.pyplot as plt

h_G = 2
L = 5

x = np.linspace(0,L, num=20)

def h_0(x):
    return -h_G*x/L + 2*h_G

plt.plot(x, h_0(x))
plt.savefig("figures/test1")