import numpy as np
from solvers.solvers import IceSheetSolver
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from matplotlib import animation
import glob

figures_folder = "figures/"
animation_folder = "steady_animation_frames/"
os.makedirs(figures_folder, exist_ok=True)
os.makedirs(animation_folder, exist_ok=True)

plt.rcParams["font.family"] = "Latin Modern Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Latin Modern Roman"
plt.rcParams["mathtext.it"] = "Latin Modern Roman:italic"
plt.rcParams["mathtext.bf"] = "Latin Modern Roman:bold"

print("Computing steady state plot...")

N = 100
x_G_0 = 0.1
nu = 80
num_microsteps = 500
t_final = 2
L = 150
D = 0

# initial state:
h_0 = 0.7*(1-0.9*np.linspace(0,1,num=N))

folder = "figures/"
os.makedirs(folder, exist_ok=True)

x_bed = np.linspace(0,20, num=2000)
b_bed = x_bed

system = IceSheetSolver(x_bed, b_bed, x_G_0, h_0, D, nu, L, num_microsteps, t_final)
system.compute_solution(compute_shelf=False)
x, h = system.x_tnsr, system.h_tnsr

print("Generating frames...")
for i in tqdm(range(h.shape[-1]-1)):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4.8,3.2), dpi=400)
    h_pred = h[:, i]
    x_pred = x[:, i]
    
    ax.plot(x[:,-1], (1-x[:,-1]), label="Analytical \nsteady state", c="blue", alpha=1, linestyle="--", linewidth=1)
    ax.plot(x_pred, h_pred, label="Numerical", c="orange", linewidth=2)
    plt.legend(prop={"family":"Latin Modern Roman", "size":12})
    ax.set_xlabel(r"$x$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    ax.set_ylabel(r"$h$ (dimensionless)", fontname = "Latin Modern Roman", fontsize=14)
    ax.set_title("Time evolution of sheet", fontname="Latin Modern Roman", fontsize=14)
    plt.savefig(animation_folder + f"full_frame_{i:03d}")
    plt.close()
    
print("Creating animation...")
filenames = sorted(glob.glob(animation_folder + "/full_frame_*.png"))

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
ani.save(figures_folder + "full_steady_solution.gif", writer="pillow", dpi=200)

print("Complete.")