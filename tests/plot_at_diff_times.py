import numpy as np
from solvers.solvers import IceSheetSolver
import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

N = 100
x_G_0 = 0.1
num_microsteps = 500
D = 0.4
t_final = 2.5
L = 150
nu = 80
domain_width = 8

def sheet_and_shelf_coords(x, h, x_shelf, H_shelf):
    x_coords = []
    h_coords = []

    # Sheet
    x_coords.append(x[:,i].reshape(-1))
    h_coords.append(h[:,i].reshape(-1))

    # Top of shelf
    x_coords.append(np.array([x_shelf[-1,i], x_shelf[0,i]]))
    h_coords.append(np.array([h_above_water,h_above_water]))

    # End of shelf
    x_coords.append(np.array([x_shelf[0,i], x_shelf[0,i]]))
    h_coords.append(np.array([-H_shelf[0,i]+h_above_water, h_above_water]))

    # Shelf
    x_coords.append(x_shelf[:,i].reshape(-1))
    h_coords.append(-H_shelf[:,i].reshape(-1)+h_above_water)

    # End bit
    x_coords.append(np.array([x_shelf[-1,i], x_shelf[-1,i]]))
    b_G = np.interp(np.array(x_shelf[-1,i]), x_bed, b_bed)
    h_coords.append(np.array([-b_G, -H_shelf[-1,i]+h_above_water]))

    return np.concatenate(x_coords, axis=0), np.concatenate(h_coords, axis=0)

# initial state:
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N))

x_bed = np.linspace(0, 20, num=2000)
#b_bed = 1.3*x_bed
b_bed = 0.2*np.sin(8*x_bed)+1.3*x_bed

system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)
system.compute_solution()
x, h, x_shelf, H_shelf = system.x_tnsr, system.h_tnsr, system.x_shelf_tnsr, system.H_shelf_tnsr

print("Producing plot at different times...")
#x, h, x_shelf, H_shelf = numerical_fv_implicit_general_b(h_0, x_bed, b_bed, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, a, L)

fig, ax = plt.subplots(constrained_layout=True, figsize=(8,2.9), dpi=400)
ax.set_title("Time evolution of ice sheet and shelf", fontname = "Latin Modern Roman", fontsize=16)
#plt.axis('equal')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, domain_width)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(r"$x$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
ax.set_ylabel(r"$z$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
#ax.plot([0,domain_width],[0,-A*domain_width], c='black', label='bedrock')
ax.plot(x_bed, -b_bed, c='black', label='bedrock')
#ax.plot(x_bed, -b_bed_test, c='black', label='bedrock_test')
ax.plot([0,domain_width], [0,0], c="blue", alpha=0.2, label='water line')

frames_to_plot = [0, 10, 22, 58, 99]
alphas = [0.8, 0.9, 0.5, 0.8, 0.8]
#alphas = np.linspace(0.8,0.8, num=len(frames_to_plot))


colours = ["red", "orange", "orange", "green", "blue"]

for j in range(len(frames_to_plot)):
    i = frames_to_plot[j]
    h_above_water = h[-1,i]
    label = f"t = {(i*t_final/system.num_steps):.1f}"
    x_coords, h_coords = sheet_and_shelf_coords(x, h, x_shelf, H_shelf)
    ax.plot(x_coords, h_coords, alpha = alphas[j], label=label, c=colours[j])

folder = "figures/"
os.makedirs(folder, exist_ok=True)

plt.legend()
plt.savefig(folder + "plot_at_diff_times.png")
print("Complete.")