import cv2
import numpy as np
import time

# Load image and convert to grayscale
original_img_color = cv2.imread('data/bonn.jpg')
gray_img = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)



print("\n--- a) Calculating Integral Image ---")
def calculate_integral_image(img):
    height, width = img.shape
    integral = np.zeros((height, width), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            integral[i, j] = img[i, j]

            if i > 0:
                integral[i, j] += integral[i - 1, j]
            if j > 0:
                integral[i, j] += integral[i, j - 1]
            if i > 0 and j > 0:
                integral[i, j] -= integral[i - 1, j - 1]
    return integral

# Calculate integral image
integral_img = calculate_integral_image(gray_img)
print(f"Integral image size: {integral_img.shape}")




print("\n--- b) Computing Mean Using Integral Image ---")
def mean_using_integral(integral, top_left, bottom_right):
    """
    Time Complexity: O(1) per query after O(H*W) preprocessing
    """
    i1, j1 = top_left
    i2, j2 = bottom_right

    region_sum = (integral[i2, j2] -
                  integral[i1 - 1, j2] -      # Subtract top
                  integral[i2, j1 - 1] +      # Subtract left
                  integral[i1 - 1, j1 - 1])   # Add back top-left

    num_pixels = (i2 - i1 + 1) * (j2 - j1 + 1)  # Correct dimensions
    return region_sum / num_pixels

# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right)
print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")




print("\n--- c) Computing Mean by Direct Summation ---")
def mean_by_direct_sum(img, top_left, bottom_right):
    """
    Time Complexity: O(w * h) where w and h are region dimensions
    """
    i1, j1 = top_left
    i2, j2 = bottom_right

    # Extract region
    region = img[i1:i2 + 1, j1:j2 + 1]

    # Calculate mean
    return np.mean(region)


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)
print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")





print("\n--- d) Computational Complexity Analysis ---")

# Benchmark parameters
iterations = 1000

print(f"\nBenchmarking with {iterations} iterations...\n")

# Benchmark integral image method
start_time = time.perf_counter()
for _ in range(iterations):
    result_integral = mean_using_integral(integral_img, top_left, bottom_right)
time_integral = time.perf_counter() - start_time

# Benchmark direct summation method
start_time = time.perf_counter()
for _ in range(iterations):
    result_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)
time_direct = time.perf_counter() - start_time

# Display results
print(f"\n{'Method':<35} {'Time Complexity':<20} {'Avg Time (ms)'}")
print("-" * 70)
print(f"{'Integral Image Method':<35} {'O(1)':<20} {time_integral / iterations * 1000:.6f}")
print(f"{'Direct Summation Method':<35} {'O(w * h)':<20} {time_direct / iterations * 1000:.6f}")

print(f"\n{'Performance Improvement:':<35} {time_direct / time_integral:.2f}x faster with integral image")

print(f"\n{'Verification:'}")
print(f"  Mean gray value (Integral): {result_integral:.2f}")
print(f"  Mean gray value (Direct): {result_direct:.2f}")
print(f"  Difference: {abs(result_integral - result_direct):.6f}")