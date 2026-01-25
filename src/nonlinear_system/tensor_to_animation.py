import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

print("Running tensor_to_animation.py...")

x = np.load("saved_objects/nonlinear_x.npy")
t = np.load("saved_objects/nonlinear_t.npy")
h_num_fd = np.load("saved_objects/ml_solution.npy")
h_num_fd = h_num_fd.reshape(h_num_fd.shape[0],h_num_fd.shape[1],1)
h_num_fv = np.load("saved_objects/numerical_solution_fd.npy")
h_num_fv = h_num_fv.reshape(h_num_fv.shape[0],h_num_fv.shape[1],1)

print("Generating frames...")
for i in range(h_num_fd.shape[1]):
    plt.figure(figsize=(10,7))
    plt.title("Numerical Solution for Nonlinear Diffusion Equation")
    plt.plot(x[:,0,0], h_num_fd[:,i,0], c='red', linestyle="dotted", label="ml-sol")
    plt.plot(x[:,0,0], h_num_fv[:,i,0], c='blue', linestyle="dashdot", label="numerical-fv")
    plt.legend()
    plt.xlim(min(x[:,0,0]), max(x[:,0,0]))
    plt.ylim(0, 7)
    plt.savefig(f"animation_frames/numerical_frame_{i:03d}")
    plt.close()
print("Complete.")

print("Creating animation...")
filenames = sorted(glob.glob("animation_frames/numerical_frame_*.png"))  # noqa: PTH207

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
ani.save("figures/solutions.gif", writer="pillow", dpi=200)

print("Complete.")