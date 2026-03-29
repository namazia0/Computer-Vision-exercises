import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree

'''
This code implements the Iterative Closest Point (ICP) algorithm to align a set of landmarks (warped) to the edges of an image (elephant).
'''

warped = np.loadtxt('rat.txt')
img = cv2.imread("rat.webp", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.scatter(warped[:, 0], warped[:, 1], c="red", s=25)
plt.plot(warped[:, 0], warped[:, 1], c="red", linewidth=1)
plt.axis("off")
plt.savefig('output/warped.png', bbox_inches='tight', pad_inches=0)

# Smooth image
img = cv2.GaussianBlur(img, (5, 5), 0)

# Run Canny edge detector
edges = cv2.Canny(img, threshold1=50, threshold2=100)
# edges = cv2.Canny(img, threshold1=100, threshold2=200)

# Invert edge map: edges must be zero for distanceTransform
edges_inv = 255 - edges

# Distance transform
dist = cv2.distanceTransform(edges_inv, cv2.DIST_L2, maskSize=5)

# Normalize for visualization
dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

plt.figure(figsize=(8, 8))
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.title("Canny Edge Map")
plt.savefig('output/canny.png')

plt.figure(figsize=(8, 8))
plt.imshow(dist_norm, cmap="magma")
plt.colorbar(label="Normalized distance to nearest edge")
plt.axis("off")
plt.title("Distance Transform of Edge Map")
plt.savefig('output/dist_transform.png')


def extract_edge_points(edge_img):
    """
    edge_img: binary edge image (255 = edge)
    returns (M, 2) array of (x, y) edge coordinates
    """
    ys, xs = np.where(edge_img > 0)
    return np.stack([xs, ys], axis=1)


def closest_edge_points(landmarks, kdtree, edge_points):
    """
    For each landmark, find closest edge point
    """
    distances, indices = kdtree.query(landmarks)
    return edge_points[indices], distances


def similarity_procrustes(X, Y):
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    Xc = X - mu_X
    Yc = Y - mu_Y

    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    s = np.trace(np.diag(S)) / np.sum(Xc ** 2)
    t = mu_Y - s * (R @ mu_X)

    return s, R, t


def icp_to_edges(landmarks,
                 edge_points,
                 kdtree,
                 max_iters=50,
                 tol=1e-3):
    X = landmarks.copy()
    prev_error = np.inf

    for it in range(max_iters):
        Y, dists = closest_edge_points(X, kdtree, edge_points)

        s, R, t = similarity_procrustes(X, Y)
        X_new = s * (X @ R.T) + t

        error = np.mean(dists)
        if abs(prev_error - error) < tol:
            print(f"Converged at iteration {it}")
            break

        X = X_new
        prev_error = error

    return X


edge_points = extract_edge_points(edges)
kdtree = cKDTree(edge_points)

aligned_landmarks = icp_to_edges(
    warped,
    edge_points,
    kdtree
)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.scatter(aligned_landmarks[:, 0], aligned_landmarks[:, 1],
            c="red", s=25)
plt.plot(aligned_landmarks[:, 0], aligned_landmarks[:, 1],
         c="red", linewidth=1)
plt.axis("off")
plt.title("Elephant image with re-aligned landmarks")
plt.savefig('output/realign.png')