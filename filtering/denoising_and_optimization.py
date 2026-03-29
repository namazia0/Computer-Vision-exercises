import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt


# Load image
original_img_color = cv2.imread('data/bonn.jpg')
original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)

# Load noisy image
noisy_img = cv2.imread('data/bonn_noisy.jpg')
noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
noisy_img_float_01 = noisy_img.astype(np.float32) / 255.0

psnr_noisy = peak_signal_noise_ratio(original_img_gray, noisy_img)
print(f"PSNR of HIGHLY Noisy Image (Mixed Noise): {psnr_noisy:.2f} dB")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_img_gray, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image (Mixed Noise PSNR: {psnr_noisy:.2f} dB)')
plt.axis('off')

plt.tight_layout()
plt.show()




# Custom Filter Definitions
def custom_gaussian_filter(image, kernel_size, sigma):
    """Custom Gaussian Filter (Convolution from scratch)"""
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    output_image = np.zeros_like(image, dtype=np.float32)
    ax = np.arange(-pad_width, pad_width + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel /= np.sum(kernel)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output_image[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)
    return output_image


def calculate_median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)

    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]
    else:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

def custom_median_filter(image, kernel_size):
    """Custom Median Filter (Median calculation from scratch)"""
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    output_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output_image[i, j] = calculate_median(padded_image[i:i + kernel_size, j:j + kernel_size].flatten())
    return output_image


def custom_bilateral_filter(image, d, sigma_color, sigma_space):
    """Custom Bilateral Filter"""
    pad_width = d // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    output_image = np.zeros_like(image, dtype=np.float32)

    # sigma_color is provided in 0-1 float range, matching input image range
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            center_pixel = padded_image[i + pad_width, j + pad_width]
            weight_sum = 0
            filtered_pixel_value = 0

            for k in range(d):
                for l in range(d):
                    current_pixel = padded_image[i + k, j + l]

                    # Domain Kernel
                    dist_sq = (k - pad_width) ** 2 + (l - pad_width) ** 2
                    spatial_weight = np.exp(-dist_sq / (2 * sigma_space ** 2))

                    # Range Kernel
                    intensity_diff_sq = (current_pixel - center_pixel) ** 2
                    range_weight = np.exp(-intensity_diff_sq / (2 * sigma_color ** 2))

                    combined_weight = spatial_weight * range_weight
                    weight_sum += combined_weight
                    filtered_pixel_value += combined_weight * current_pixel

            output_image[i, j] = filtered_pixel_value / weight_sum
    return output_image




# Filter Application - Using Default/Non-Optimal Parameters
print("\n--- Filter Application ---")

# Default Parameters
K_DEFAULT = 7           # Kernel size for Gaussian and Median filters (7x7)
S_DEFAULT = 2.0         # Sigma for Gaussian filter
D_DEFAULT = 9           # Diameter for Bilateral filter (9x9 neighborhood)
SC_DEFAULT = 100        # Sigma Color (sigma_r) for cv2.bilateralFilter - in [0-255] range
SS_DEFAULT = 75         # Sigma Space (sigma_d) for cv2.bilateralFilter

# -------------------------- a) Gaussian Filter --------------------------
print("a) Applying Gaussian Filter...")
# CV2
denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img, (K_DEFAULT, K_DEFAULT), S_DEFAULT)
psnr_gaussian_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_gaussian_cv2)
# Custom
gaussian_custom_float = custom_gaussian_filter(noisy_img_float_01, kernel_size=K_DEFAULT, sigma=S_DEFAULT)
denoised_gaussian_custom = (gaussian_custom_float * 255).astype(np.uint8)
psnr_gaussian_custom = peak_signal_noise_ratio(original_img_gray, denoised_gaussian_custom)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image (PSNR: {psnr_noisy:.2f} dB)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(denoised_gaussian_cv2, cmap='gray')
plt.title(f'CV2 Gaussian (PSNR: {psnr_gaussian_cv2:.2f} dB)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(denoised_gaussian_custom, cmap='gray')
plt.title(f'Custom Gaussian (PSNR: {psnr_gaussian_custom:.2f} dB)')
plt.axis('off')
plt.suptitle('a) Gaussian Filter Results (CV2 vs. Custom)', fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------- b) Median Filter --------------------------
print("b) Applying Median Filter...")
# CV2
denoised_median_cv2 = cv2.medianBlur(noisy_img, K_DEFAULT)
psnr_median_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_median_cv2)
# Custom
median_custom_float = custom_median_filter(noisy_img_float_01, kernel_size=K_DEFAULT)
denoised_median_custom = (median_custom_float * 255).astype(np.uint8)
psnr_median_custom = peak_signal_noise_ratio(original_img_gray, denoised_median_custom)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image (PSNR: {psnr_noisy:.2f} dB)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(denoised_median_cv2, cmap='gray')
plt.title(f'CV2 Median (PSNR: {psnr_median_cv2:.2f} dB)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(denoised_median_custom, cmap='gray')
plt.title(f'Custom Median (PSNR: {psnr_median_custom:.2f} dB)')
plt.axis('off')
plt.suptitle('b) Median Filter Results (CV2 vs. Custom)', fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------- c) Bilateral Filter --------------------------
print("c) Applying Bilateral Filter...")
# CV2
denoised_bilateral_cv2 = cv2.bilateralFilter(noisy_img, D_DEFAULT, SC_DEFAULT, SS_DEFAULT)
psnr_bilateral_cv2 = peak_signal_noise_ratio(original_img_gray, denoised_bilateral_cv2)
# Custom (sigma_color scaled for 0-1 input image range)
bilateral_custom_float = custom_bilateral_filter(noisy_img_float_01, d=D_DEFAULT, sigma_color=SC_DEFAULT / 255.0,
                                                 sigma_space=SS_DEFAULT)
denoised_bilateral_custom = (bilateral_custom_float * 255).astype(np.uint8)
psnr_bilateral_custom = peak_signal_noise_ratio(original_img_gray, denoised_bilateral_custom)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image (PSNR: {psnr_noisy:.2f} dB)')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(denoised_bilateral_cv2, cmap='gray')
plt.title(f'CV2 Bilateral (PSNR: {psnr_bilateral_cv2:.2f} dB)')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(denoised_bilateral_custom, cmap='gray')
plt.title(f'Custom Bilateral (PSNR: {psnr_bilateral_custom:.2f} dB)')
plt.axis('off')
plt.suptitle('c) Bilateral Filter Results (CV2 vs. Custom)', fontweight='bold')
plt.tight_layout()
plt.show()




# Performance Comparison
print("\n--- d) Performance Comparison ---")
print("Comparing all three filters using default parameters:\n")

print(f"Gaussian Filter  - PSNR: {psnr_gaussian_cv2:.2f} dB (kernel_size={K_DEFAULT}, sigma={S_DEFAULT})")
print(f"Median Filter    - PSNR: {psnr_median_cv2:.2f} dB (kernel_size={K_DEFAULT})")
print(f"Bilateral Filter - PSNR: {psnr_bilateral_cv2:.2f} dB (d={D_DEFAULT}, sigma_color={SC_DEFAULT}, sigma_space={SS_DEFAULT})")

# Determine which filter performs best
filter_psnrs = {
    'Gaussian': psnr_gaussian_cv2,
    'Median': psnr_median_cv2,
    'Bilateral': psnr_bilateral_cv2
}
best_filter = max(filter_psnrs, key=filter_psnrs.get)
print(f"\n** Best performing filter with default parameters: {best_filter} (PSNR: {filter_psnrs[best_filter]:.2f} dB) **\n")

# Display side-by-side comparison
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'Noisy Image\nPSNR: {psnr_noisy:.2f} dB')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(denoised_gaussian_cv2, cmap='gray')
plt.title(f'Gaussian Filter\nPSNR: {psnr_gaussian_cv2:.2f} dB')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(denoised_median_cv2, cmap='gray')
plt.title(f'Median Filter\nPSNR: {psnr_median_cv2:.2f} dB')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(denoised_bilateral_cv2, cmap='gray')
plt.title(f'Bilateral Filter\nPSNR: {psnr_bilateral_cv2:.2f} dB')
plt.axis('off')

plt.suptitle('d) Performance Comparison: All Three Filters', fontweight='bold')
plt.tight_layout()
plt.show()





# Parameter Optimization
def run_optimization(original_img, noisy_img):
    """
    Iterates through relevant parameters for Gaussian, Median, and Bilateral filters
    to find the combination that yields the highest PSNR using cv2 functions.
    """
    print("\n--- e) Parameter Optimization Started ---")

    optimization_results = {}

    # Gaussian Filter Optimization
    best_gaussian_psnr = -1
    best_gaussian_params = {}

    kernel_sizes = [3, 5, 7, 9, 11]
    sigma_start = 0.1
    sigma_end = 6.0

    for K in kernel_sizes:
        for S in np.arange(sigma_start, sigma_end, 0.05):
            denoised_img = cv2.GaussianBlur(noisy_img, (K, K), S)
            psnr = peak_signal_noise_ratio(original_img, denoised_img)
            if psnr > best_gaussian_psnr:
                best_gaussian_psnr = psnr
                best_gaussian_params = {'kernel_size': K, 'sigma': S}

    optimization_results['Gaussian'] = {'best_psnr': best_gaussian_psnr, 'optimal_params': best_gaussian_params}
    print(f"-> Gaussian Optimal PSNR: {best_gaussian_psnr:.2f} dB at {best_gaussian_params}")


    # Median Filter Optimization
    best_median_psnr = -1
    best_median_params = {}

    kernel_sizes = [3, 5, 7, 9, 11]

    for K in kernel_sizes:
        denoised_img = cv2.medianBlur(noisy_img, K)
        psnr = peak_signal_noise_ratio(original_img, denoised_img)
        if psnr > best_median_psnr:
            best_median_psnr = psnr
            best_median_params = {'kernel_size': K}

    optimization_results['Median'] = {'best_psnr': best_median_psnr, 'optimal_params': best_median_params}
    print(f"-> Median Optimal PSNR: {best_median_psnr:.2f} dB at {best_median_params}")

    # Bilateral Filter Optimization
    best_bilateral_psnr = -1
    best_bilateral_params = {}

    d_values = [3, 5, 7, 9, 11]
    sigma_color_values_start = 70
    sigma_color_values_end = 150
    sigma_space_values_start = 1
    sigma_space_values_end = 30

    for D in d_values:
        for SC in range(sigma_color_values_start, sigma_color_values_end, 1):
            for SS in range(sigma_space_values_start, sigma_space_values_end):
                denoised_img = cv2.bilateralFilter(noisy_img, D, SC, SS)
                psnr = peak_signal_noise_ratio(original_img, denoised_img)
                if psnr > best_bilateral_psnr:
                    best_bilateral_psnr = psnr
                    best_bilateral_params = {'d': D, 'sigma_color': SC, 'sigma_space': SS}

    optimization_results['Bilateral'] = {'best_psnr': best_bilateral_psnr, 'optimal_params': best_bilateral_params}
    print(f"-> Bilateral Optimal PSNR: {best_bilateral_psnr:.2f} dB at {best_bilateral_params}")

    return optimization_results




# Execution and Display of Optimal Results

optimal_results = run_optimization(original_img_gray, noisy_img)

# These are the parameters found through optimization:
OPTIMAL_GAUSSIAN_KERNEL = optimal_results['Gaussian']['optimal_params']['kernel_size']
OPTIMAL_GAUSSIAN_SIGMA = optimal_results['Gaussian']['optimal_params']['sigma']

OPTIMAL_MEDIAN_KERNEL = optimal_results['Median']['optimal_params']['kernel_size']

OPTIMAL_BILATERAL_D = optimal_results['Bilateral']['optimal_params']['d']
OPTIMAL_BILATERAL_SIGMA_COLOR = optimal_results['Bilateral']['optimal_params']['sigma_color']
OPTIMAL_BILATERAL_SIGMA_SPACE = optimal_results['Bilateral']['optimal_params']['sigma_space']

print("\n--- Optimal Parameters Summary ---")
print(f"Gaussian: kernel_size={OPTIMAL_GAUSSIAN_KERNEL}, sigma={OPTIMAL_GAUSSIAN_SIGMA}")
print(f"Median: kernel_size={OPTIMAL_MEDIAN_KERNEL}")
print(f"Bilateral: d={OPTIMAL_BILATERAL_D}, sigma_color={OPTIMAL_BILATERAL_SIGMA_COLOR}, sigma_space={OPTIMAL_BILATERAL_SIGMA_SPACE}")




# Get Optimal Images using the parameters found
# Gaussian
optimal_gaussian = cv2.GaussianBlur(noisy_img, (OPTIMAL_GAUSSIAN_KERNEL, OPTIMAL_GAUSSIAN_KERNEL), OPTIMAL_GAUSSIAN_SIGMA)

# Median
optimal_median = cv2.medianBlur(noisy_img, OPTIMAL_MEDIAN_KERNEL)

# Bilateral
optimal_bilateral = cv2.bilateralFilter(noisy_img, OPTIMAL_BILATERAL_D, OPTIMAL_BILATERAL_SIGMA_COLOR, OPTIMAL_BILATERAL_SIGMA_SPACE)

# --- Display the Optimal Images ---
plt.figure(figsize=(16, 8))

# Original Noisy Image for context
plt.subplot(2, 2, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title(f'1. Original Noisy Image\nPSNR: {psnr_noisy:.2f} dB')
plt.axis('off')

# Optimal Gaussian
plt.subplot(2, 2, 2)
plt.imshow(optimal_gaussian, cmap='gray')
plt.title(f"2. Optimal Gaussian\nPSNR: {optimal_results['Gaussian']['best_psnr']:.2f} dB\nK={OPTIMAL_GAUSSIAN_KERNEL}, σ={OPTIMAL_GAUSSIAN_SIGMA}")
plt.axis('off')

# Optimal Median
plt.subplot(2, 2, 3)
plt.imshow(optimal_median, cmap='gray')
plt.title(f"3. Optimal Median\nPSNR: {optimal_results['Median']['best_psnr']:.2f} dB\nK={OPTIMAL_MEDIAN_KERNEL}")
plt.axis('off')

# Optimal Bilateral
plt.subplot(2, 2, 4)
plt.imshow(optimal_bilateral, cmap='gray')
plt.title(f"4. Optimal Bilateral\nPSNR: {optimal_results['Bilateral']['best_psnr']:.2f} dB\nd={OPTIMAL_BILATERAL_D}, σc={OPTIMAL_BILATERAL_SIGMA_COLOR}, σs={OPTIMAL_BILATERAL_SIGMA_SPACE}")
plt.axis('off')

plt.suptitle('e) Parameter Optimization: Best Results', fontweight='bold')
plt.tight_layout()
plt.show()
