import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

print("Running tensor_to_animation.py...")

x = np.load("saved_objects/linear_x.npy")
t = np.load("saved_objects/linear_t.npy")
h_an = np.load("saved_objects/linear_analytical_solution.npy")
h_num_ex = np.load("saved_objects/linear_numerical_solution_ex.npy")
h_num_im = np.load("saved_objects/linear_numerical_solution_im.npy")
h_num = h_num_ex.reshape(h_num_ex.shape[0], h_num_ex.shape[1], 1)


h_num_ex = h_num_ex.reshape(h_num_ex.shape[0],h_num_ex.shape[1],1)
h_num_im = h_num_im.reshape(h_num_im.shape[0],h_num_im.shape[1],1)

print("Generating frames...")
for i in range(h_an.shape[1]):
    plt.figure(figsize=(10,7))
    plt.title("Analytical vs Numerical Solution for Linear Diffusion Equation")
    plt.plot(x[:,0,0], h_an[:,i,0], c='red', linestyle="solid", label="analytical")
    plt.plot(x[:,0,0], h_num_ex[:,i,0], c='blue', linestyle="dashed", label="explicit-numerical")
    plt.plot(x[:,0,0], h_num_im[:,i,0], c='yellow', linestyle="dashdot", label="implicit-numerical")
    plt.legend()
    plt.xlim(min(x[:,0,0]), max(x[:,0,0]))
    plt.ylim(0, 7)
    plt.savefig(f"animation_frames/analytical_linear_frame_{i:03d}")
    plt.close()
print("Complete.")

print("Creating animation...")
filenames = sorted(glob.glob("animation_frames/analytical_linear_frame_*.png"))  # noqa: PTH207

fig = plt.figure()
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
ani.save("figures/linear_solutions.gif", writer="pillow", dpi=200)

print("Complete.")