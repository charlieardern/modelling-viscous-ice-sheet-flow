import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

print("Running tensor_to_animation.py...")

x = np.load("saved_objects/full_x.npy")
h = np.load("saved_objects/full_h.npy")

print("Generating frames...")
for i in range(h.shape[1]):
    plt.figure(figsize=(10,7))
    plt.title("Numerical Solution for Nonlinear Diffusion Equation")
    plt.plot(x[:,i], h[:,i], c='blue', linestyle="solid", label="numerical-fv")
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(-10, 15)
    plt.plot([x[-1,i], x[-1,i]], [-0.1*x[-1,i], 0], c='blue')
    plt.plot([0,100],[0,-10], c='black') #bedrock illustration with alpha=-0.1
    plt.plot([0,100], [0,0], c="blue", alpha=0.2)
    plt.savefig(f"animation_frames/full_numerical_frame_{i:03d}")
    plt.close()
print("Complete.")

print("Creating animation...")
filenames = sorted(glob.glob("animation_frames/full_numerical_frame_*.png"))  # noqa: PTH207

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
ani.save("figures/full_solution.gif", writer="pillow", dpi=200)

print("Complete.")