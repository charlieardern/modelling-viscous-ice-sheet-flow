import numpy as np
from solvers.solvers import numerical_fv_implicit_general_b
import os
import glob
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib import animation

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

figures_folder = "figures/"
animation_folder = "animation_frames/"
os.makedirs(figures_folder, exist_ok=True)
os.makedirs(animation_folder, exist_ok=True)

N = 100
x_G_0 = 0.1
alpha = 0.18
nu = 80
num_steps = 100
num_microsteps = 500
t_final = 320
rho_w = 1000
rho = 917
g = 9.81
a = 0.025
L = 150
domain_width = 6

q_0 = a*L
g_prime = g*(rho_w-rho)/rho_w
A = 2*alpha*np.sqrt(g/g_prime)
B = (6*nu*q_0/g)**(1/3)
C = (g/g_prime)**(1/6)
eps = 2*alpha*(g_prime/g)**(0.5)*rho_w/rho
D = 2*B*C**4/L

print(f"A: {A}, D: {D}, ε: {eps}")

print(f"scaled final time: {t_final*(a*L/(2*B**2*C**5))}")

def sheet_and_shelf_coords(x, h, x_shelf, H_shelf):
    x_coords = []
    h_coords = []
    h_above_water = h[-1,i]

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
h_0 = 0.7*(1-np.linspace(0,1,num=N)+0.002+0.1)

x_bed = np.linspace(0, 20, num=2000)
#b_bed = 1.3*x_bed
b_bed = 0.2*np.sin(8*x_bed)+1.3*x_bed

print("Solving...")
x, h, x_shelf, H_shelf = numerical_fv_implicit_general_b(h_0, x_bed, b_bed, num_microsteps, num_steps, t_final, x_G_0, g, nu, rho_w, rho, a, L)

# Non-dimensionalise x and h:
x = x/(2*B*C**4)
h = h/(B*C)
x_shelf = x_shelf/(2*B*C**4)
H_shelf = H_shelf/(B*C)

frames_to_plot = [0, 10, 25, 58, 99]
alphas = [0.8, 0.8, 0.4, 0.8, 0.8]
#alphas = np.linspace(0.8,0.8, num=len(frames_to_plot))

colours = ["red", "orange", "orange", "green", "blue"]

print("Generating frames...")
for i in tqdm(range(h.shape[-1]-1)):
    x_coords, h_coords = sheet_and_shelf_coords(x, h, x_shelf, H_shelf)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(10,5), dpi=400)
    ax.set_title("Time evolution of ice sheet and shelf", fontname = "Latin Modern Roman", fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, domain_width)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"$x$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    ax.set_ylabel(r"$z$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    ax.plot(x_bed, -b_bed, c='black', label='bedrock')
    ax.plot([0,domain_width], [0,0], c="blue", alpha=0.2, label='water line')
    ax.plot(x_coords, h_coords, c="blue")
    plt.savefig(animation_folder + f"full_frame_{i:03d}")
    plt.close()
    
print("Creating animation...")
filenames = sorted(glob.glob("animation_frames/full_frame_*.png"))

fig = plt.figure(figsize=(10,5))
plt.axis("off")

# Create a list of image artists: each element must be a list (for ArtistAnimation)
ims = []
num_frames = len(filenames)
for i, fname in enumerate(filenames):
    img = plt.imread(fname)
    im = plt.imshow(img, animated=True)
    ims.append([im])
    if i == (num_frames - 1):
        ims.extend([[im]] * 200)
fig.tight_layout()
# Create the animation: interval is in milliseconds.
ani = animation.ArtistAnimation(
    fig,
    ims,  # type: ignore # noqa: PGH003
    interval=100,
    blit=True,
    repeat_delay=3000,
)
# Save the animation (requires ffmpeg or ImageMagick)
ani.save(figures_folder + "full_solution.gif", writer="pillow", dpi=200)

print("Complete.")
