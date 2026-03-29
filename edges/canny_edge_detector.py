import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Smoothing to reduce noise
def gaussian_smoothing(img, sigma):
    return cv2.GaussianBlur(img, (3, 3), sigma)

# Find strength and direction of each pixel
def compute_gradients(img):
    gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    abs_gx = cv2.convertScaleAbs(gx)
    abs_gy = cv2.convertScaleAbs(gy)

    mag = cv2.addWeighted(abs_gx, 0.5, abs_gy, 0.5, 0).astype(np.float32)
    ang = np.arctan2(gy.astype(np.float32), gx.astype(np.float32))
    return mag, ang

# Thinning process, only keeping the peaks
def nonmax_suppression(mag, ang):
    H, W = mag.shape
    Z = np.zeros((H, W), np.float32)
    angle = ang * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 255
            r = 255
            # Case 1: Angle is ~0° or ~180° (Horizontal gradient)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = mag[i, j + 1] # Neighbor to the East
                r = mag[i, j - 1] # Neighbor to the West
            # Case 2: Angle is ~45° (NE/SW gradient)
            elif (22.5 <= angle[i, j] < 67.5):
                q = mag[i + 1, j - 1] # Neighbor to the South-West
                r = mag[i - 1, j + 1] # Neighbor to the North-East
            # Case 3: Angle is ~90° (Vertical gradient)
            elif (67.5 <= angle[i, j] < 112.5):
                q = mag[i + 1, j] # Neighbor to the South
                r = mag[i - 1, j] # Neighbor to the North
            # Case 4: Angle is ~135° (NW/SE gradient)
            elif (112.5 <= angle[i, j] < 157.5):
                q = mag[i - 1, j - 1] # Neighbor to the North-West
                r = mag[i + 1, j + 1] # Neighbor to the South-East

            if (mag[i, j] >= q) and (mag[i, j] >= r):
                Z[i, j] = mag[i, j]

    return Z

# classify thinned edges into three categories
def double_threshold(nms, low, high):
    strong, weak = 255, 75
    res = np.zeros_like(nms)
    res[nms >= high] = strong
    res[(nms >= low) & (nms < high)] = weak
    return res, weak, strong

# connect-the-dot steps to remove isolated noisy steps
def hysteresis(edge_map, weak, strong):
    H, W = edge_map.shape
    q = deque(np.argwhere(edge_map == strong))

    while q:
        i, j = q.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (0 <= ni < H and 0 <= nj < W and
                        edge_map[ni, nj] == weak):
                    edge_map[ni, nj] = strong
                    q.append((ni, nj))

    edge_map[edge_map != strong] = 0
    edge_map[edge_map == strong] = 1
    return edge_map


def compute_metrics(manual_edges, cv_edges):
    mad = np.mean(np.abs(manual_edges - cv_edges))

    tp = np.sum((manual_edges == 1) & (cv_edges == 1))
    fp = np.sum((manual_edges == 1) & (cv_edges == 0))
    fn = np.sum((manual_edges == 0) & (cv_edges == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return mad, precision, recall, f1


# Load
img = cv2.imread('data/bonn.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Smooth
blur = gaussian_smoothing(img, sigma=0.0)

# Gradients
mag, ang = compute_gradients(blur)

# Non-Maximum Suppression (NMS)
nms = nonmax_suppression(mag, ang)

# Double thresholding
edges_th, weak, strong = double_threshold(nms, low=35, high=65)

# Hysteresis
edges_manual = hysteresis(edges_th.copy(), weak, strong).astype(np.float32)

# Built-in Canny for comparison
edges_cv = cv2.Canny((img * 255).astype(np.uint8), 100, 200) / 255.0

# Metrics
mad, precision, recall, f1 = compute_metrics(edges_manual, edges_cv)
print(f"MAD(Manual vs OpenCV) = {mad:.4f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img, cmap='gray'); ax[0].set_title('Original')
ax[1].imshow(edges_manual, cmap='gray'); ax[1].set_title('Manual Canny')
ax[2].imshow(edges_cv, cmap='gray'); ax[2].set_title('OpenCV Canny')

# save the figure
plt.savefig('output/canny_comparison.png', dpi=300)

for a in ax: a.axis('off')
plt.tight_layout()
plt.show()
