import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

print("Running tensor_to_animation.py...")

x = np.load("saved_objects/nonlinear_x.npy")
t = np.load("saved_objects/nonlinear_t.npy")
h_num = np.load("saved_objects/numerical_solution.npy")
h_num = h_num.reshape(h_num.shape[0],h_num.shape[1],1)

print("Generating frames...")
for i in range(h_num.shape[1]):
    plt.figure(figsize=(10,7))
    plt.title("Numerical Solution for Nonlinear Diffusion Equation")
    plt.plot(x[:,0,0], h_num[:,i,0], c='red', linestyle="solid", label="analytical")
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