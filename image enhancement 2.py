import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Baca gambar berwarna
image_color = imageio.imread("C:\\gambar\\bunga.jpg")
# Konversi ke grayscale
image_gray = np.dot(image_color[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

# Low-pass filter (Gaussian smoothing)
low_pass_gray = ndimage.gaussian_filter(image_gray, sigma=2)
low_pass_color = ndimage.gaussian_filter(image_color, sigma=2)

# High-pass filter (Laplacian)
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]])
high_pass_gray = ndimage.convolve(image_gray, laplacian_kernel)
high_pass_color = np.zeros_like(image_color)
for i in range(3):  # Untuk setiap saluran RGB
    high_pass_color[:, :, i] = ndimage.convolve(image_color[:, :, i], laplacian_kernel)

# High-boost filter
boost_factor = 1.5  # Faktor penajaman
high_boost_gray = image_gray + boost_factor * high_pass_gray
high_boost_gray = np.clip(high_boost_gray, 0, 255).astype(np.uint8)

high_boost_color = image_color + boost_factor * high_pass_color
high_boost_color = np.clip(high_boost_color, 0, 255).astype(np.uint8)

# Plot hasilnya
fig, axes = plt.subplots(3, 4, figsize=(12, 8))

# Original Images
axes[0, 0].imshow(image_gray, cmap='gray')
axes[0, 0].set_title('Original Grayscale')
axes[0, 1].imshow(image_color)
axes[0, 1].set_title('Original Color')

# Low-pass filter
axes[1, 0].imshow(low_pass_gray, cmap='gray')
axes[1, 0].set_title('Low-pass Grayscale')
axes[1, 1].imshow(low_pass_color)
axes[1, 1].set_title('Low-pass Color')

# High-pass filter
axes[2, 0].imshow(high_pass_gray, cmap='gray')
axes[2, 0].set_title('High-pass Grayscale')
axes[2, 1].imshow(np.clip(high_pass_color, 0, 255))
axes[2, 1].set_title('High-pass Color')

# High-boost filter
axes[0, 2].imshow(high_boost_gray, cmap='gray')
axes[0, 2].set_title('High-boost Grayscale')
axes[0, 3].imshow(high_boost_color)
axes[0, 3].set_title('High-boost Color')

# Hapus label sumbu
for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
