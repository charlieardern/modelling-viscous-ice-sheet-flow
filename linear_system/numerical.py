import numpy as np

print("Running numerical.py...")
h_G = 2.5
L = 4
a = 0.02
k = 0.01
num_terms = 10
w = 1

pi = np.pi

x = np.linspace(0,L, num=20).reshape(-1, 1, 1)
t = np.linspace(0,100, num=100).reshape(1,-1, 1)
n = np.arange(0,num_terms, 1).reshape(1,1,-1)

A = np.eye(t.shape[1], k=1)+np.eye(t.shape[1], k=-1)-2*np.eye(t.shape[1])
A = A*k/w**2
A[0,0] = -1
A[-1,-1] = -3

B = a*np.ones(t.shape[1])
B[-1]+= 2*k/w**2