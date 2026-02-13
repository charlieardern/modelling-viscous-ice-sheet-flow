import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import glob

print("Running tensor_to_animation.py...")

x = np.load("saved_objects/full_x.npy")
h = np.load("saved_objects/full_h.npy")
x_shelf = np.load("saved_objects/full_x_shelf.npy")
H_shelf = np.load("saved_objects/full_H_shelf.npy")

bed_alpha = 0.16
domain_width=160

print("Generating frames...")
for i in range(h.shape[1]-1):
    h_above_water = h[-1,i]
    plt.figure(figsize=(10,7))
    plt.title("Time evolution of ice sheet and shelf")
    plt.plot(x[:,i], h[:,i], c='blue', linestyle="solid", label="sheet")
    #plt.plot(x_shelf[:,i], -H_shelf[:,i], c='red', linestyle='solid')
    plt.plot(x_shelf[:,i], -H_shelf[:,i]+h_above_water, c='red', linestyle='solid', label='shelf')
    plt.plot([x_shelf[0,i], x_shelf[0,i]], [-H_shelf[0,i]+h_above_water, h_above_water], c='red')
    plt.plot([x_shelf[-1,i], x_shelf[0,i]], [h_above_water,h_above_water], c='red')
    
    plt.xlim(0, domain_width)
    plt.ylim(-bed_alpha*domain_width, 15)
    #plt.plot([x[-1,i], x[-1,i]], [-0.1*x[-1,i], 0], c='blue')
    plt.plot([0,domain_width],[0,-bed_alpha*domain_width], c='black', label='bedrock') #bedrock illustration
    plt.plot([0,domain_width], [0,0], c="blue", alpha=0.2, label='water line')
    plt.legend()
    plt.savefig(f"animation_frames/full_numerical_frame_{i:03d}")
    plt.close()
print("Complete.")

print(x_shelf[:,-1])
print(H_shelf[:,-1])

plt.figure()
plt.plot(x[:,-1], h[:,-1])
plt.savefig("figures/test.png")

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