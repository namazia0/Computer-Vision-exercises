import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
In this exercise, we use Normalized Cross-Correlation to find corresponding points between two stereo image
It computes a disparity map from a pair of rectified stereo images by comparing local patches and finding the best match in the right image for each pixel in the left image. 
'''

WINDOW_SIZE = 11       # NCC patch size
MAX_DISPARITY = 64     # Maximum search range
PREFILTER_KSIZE = (5, 5)
if WINDOW_SIZE % 2 == 0:
    WINDOW_SIZE += 1
if MAX_DISPARITY % 16 != 0:
    MAX_DISPARITY = (MAX_DISPARITY // 16 + 1) * 16
half_window = WINDOW_SIZE // 2


def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity):
    """
    Compute a dense disparity map using Normalized Cross-Correlation (NCC).
    
    Arguments:
        left_image, right_image : input grayscale stereo pair
        window_size             : size of the correlation window
        max_disparity           : maximum horizontal shift to consider

    Returns:
        disparity_map : computed disparity for each pixel (float32)
    """
    H, W = left_image.shape
    half = window_size // 2
    disparity_map = np.zeros((H, W), np.float32)

    for y in range(half, H - half):
        for x in range(half, W - half):
            left_patch = left_image[y-half:y+half+1, x-half:x+half+1]
            left_mean = np.mean(left_patch)
            left_norm = left_patch - left_mean
            left_denom = np.sqrt(np.sum(left_norm**2))
            if left_denom < 1e-6:
                continue

            best_ncc = -1.0
            best_d = 0
            ncc_scores = np.zeros(max_disparity, np.float32)

            for d in range(max_disparity):
                xr = x - d
                if xr - half < 0:
                    break
                right_patch = right_image[y-half:y+half+1, xr-half:xr+half+1]
                right_mean = np.mean(right_patch)
                right_norm = right_patch - right_mean
                right_denom = np.sqrt(np.sum(right_norm**2))
                denom = left_denom * right_denom
                ncc = np.sum(left_norm * right_norm) / denom if denom > 1e-6 else 0.0
                ncc_scores[d] = ncc
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_d = d

            # Sub-pixel refinement
            if 0 < best_d < max_disparity - 1:
                y1, y2, y3 = ncc_scores[best_d-1], ncc_scores[best_d], ncc_scores[best_d+1]
                denom = 2*(y1 - 2*y2 + y3)
                if abs(denom) > 1e-6:
                    delta = (y1 - y3) / denom
                    if abs(delta) < 0.5:
                        disparity_map[y, x] = best_d + delta
                        continue
            disparity_map[y, x] = best_d
    return disparity_map


def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    """
    if mask is not None:
        return float(np.mean(np.abs(a[mask] - b[mask])))
    return float(np.mean(np.abs(a - b)))


# Load stereo images
left_img = cv2.imread('data/left.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
right_img = cv2.imread('data/right.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
H, W = left_img.shape
left_blur = cv2.GaussianBlur(left_img, PREFILTER_KSIZE, 0)
right_blur = cv2.GaussianBlur(right_img, PREFILTER_KSIZE, 0)
 
# Manual NCC
manual_map = compute_manual_ncc_map(left_blur, right_blur, WINDOW_SIZE, MAX_DISPARITY)

# Benchmark (StereoBM)
stereo = cv2.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=WINDOW_SIZE)
bm_map = stereo.compute(left_img.astype(np.uint8), right_img.astype(np.uint8)).astype(np.float32) / 16.0

# Visualization
manual_vis = cv2.normalize(manual_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
bm_vis = cv2.normalize(bm_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure(figsize=(10, 8))
plt.subplot(2,2,1); plt.imshow(left_img, cmap='gray'); plt.title("Left Image"); plt.axis('off')
plt.subplot(2,2,2); plt.imshow(right_img, cmap='gray'); plt.title("Right Image"); plt.axis('off')
plt.subplot(2,2,3); plt.imshow(manual_vis, cmap='gray'); plt.title("Manual NCC"); plt.axis('off')
plt.subplot(2,2,4); plt.imshow(bm_vis, cmap='gray'); plt.title("StereoBM"); plt.axis('off')
plt.tight_layout(); plt.show()

# save visualizations
plt.imsave('output/manual_ncc.png', manual_vis, cmap='gray')
plt.imsave('output/stereo_bm.png', bm_vis, cmap='gray')

# Quantitative comparison
mask_valid = bm_map > 0
mae_value = compute_mae(manual_map, bm_map, mask_valid)
print(f"MAE (manual vs StereoBM) = {mae_value:.4f}")