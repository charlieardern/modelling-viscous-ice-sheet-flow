import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

print("Running tensor_to_animation.py...")

x = np.load("saved_objects/linear_analytical_x.npy")
t = np.load("saved_objects/linear_analytical_t.npy")
h = np.load("saved_objects/linear_analytical_solution.npy")

print("Generating frames...")
for i in range(h.shape[1]):
    plt.figure()
    plt.plot(x[:,0,0], h[:,i,0])
    plt.xlim(min(x[:,0,0]), max(x[:,0,0]))
    plt.ylim(0, 7)
    plt.savefig(f"animation_frames/analytical_linear_frame_{i:03d}")

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
ani.save("figures/linear_analytical_solution.gif", writer="pillow", dpi=200)

print("Complete")