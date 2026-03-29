import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted spectrum
    - The magnitude
    - The phase
    """
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(F)
    phase = np.angle(F)
    return F, mag, phase


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    """
    # Combine given magnitude and phase then inverse FFT
    F = mag * np.exp(1j * phase)
    img_rec = np.fft.ifft2(np.fft.ifftshift(F))
    img_rec = np.real(img_rec)
    # Normalize to [0, 1]
    img_rec = (img_rec - img_rec.min()) / (img_rec.max() - img_rec.min())
    return img_rec


def compute_mad(a, b):
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    return float(np.mean(np.abs(a - b)))


# Load Images
img1 = cv2.imread('data/1.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
img2 = cv2.imread('data/2.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
h = min(img1.shape[0], img2.shape[0])
w = min(img1.shape[1], img2.shape[1])
img1, img2 = img1[:h, :w], img2[:h, :w]

# FFT, Magnitude, Phase
F1, mag1, pha1 = compute_fft(img1)
F2, mag2, pha2 = compute_fft(img2)

# Swap Components
rec12 = reconstruct_from_mag_phase(mag1, pha2)  # Mag1 + Phase2
rec21 = reconstruct_from_mag_phase(mag2, pha1)  # Mag2 + Phase1

# Save Reconstructed Images
cv2.imwrite('output/reconstructed_mag1_phase2.png', (rec12 * 255).astype(np.uint8))
cv2.imwrite('output/reconstructed_mag2_phase1.png', (rec21 * 255).astype(np.uint8))

# Mean Absolute Difference
mad_i1_rec12 = compute_mad(img1, rec12)
mad_i2_rec21 = compute_mad(img2, rec21)
mad_i1_rec21 = compute_mad(img1, rec21)
mad_i2_rec12 = compute_mad(img2, rec12)
print("MAD Results:")
print(f"MAD(Image1, Mag1+Phase2) = {mad_i1_rec12:.6f}")
print(f"MAD(Image2, Mag2+Phase1) = {mad_i2_rec21:.6f}")
print(f"MAD(Image1, Mag2+Phase1) = {mad_i1_rec21:.6f}")
print(f"MAD(Image2, Mag1+Phase2) = {mad_i2_rec12:.6f}")

# Visualization
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
ax[0,0].imshow(img1, cmap='gray'); ax[0,0].set_title('Image 1')
ax[1,0].imshow(img2, cmap='gray'); ax[1,0].set_title('Image 2')
ax[0,1].imshow(np.log1p(mag1), cmap='gray'); ax[0,1].set_title('Mag 1')
ax[1,1].imshow(np.log1p(mag2), cmap='gray'); ax[1,1].set_title('Mag 2')
ax[0,2].imshow(pha1, cmap='twilight'); ax[0,2].set_title('Phase 1')
ax[1,2].imshow(pha2, cmap='twilight'); ax[1,2].set_title('Phase 2')
ax[0,3].imshow(rec12, cmap='gray'); ax[0,3].set_title('Mag1 + Phase2')
ax[1,3].imshow(rec21, cmap='gray'); ax[1,3].set_title('Mag2 + Phase1')

# save visualization
plt.tight_layout()
plt.savefig('output/fourier_analysis.png', dpi=150, bbox_inches='tight')

for a in ax.ravel(): a.axis('off')
plt.tight_layout()
plt.show()
