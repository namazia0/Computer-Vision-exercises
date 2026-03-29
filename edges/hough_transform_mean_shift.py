import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

"""
Task 2: Implement the Hough transform to detect circles in an image.
Task 3: Implement the mean shift algorithm and use it to find peaks in the Hough accumulator.
"""

def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    max_radius = min(max_radius, int(np.linalg.norm(edges.shape)))

    edges_points = np.array(np.nonzero(edges))
    h, w = edges.shape

    # Precompute
    thetas = np.arange(0, 360, theta_ssz) * np.pi / 180
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    radii = np.arange(min_radius, max_radius, r_ssz)

    # 3D accumulator array: [radius_idx, y, x]
    accumulator = np.zeros((len(radii), h, w), dtype=np.int32)

    # Vectorized voting
    for i in range(edges_points.shape[1]):    
        for r_idx, r in enumerate(radii):
            a = (edges_points[1][i] - r * cos_thetas).astype(int)
            b = (edges_points[0][i] - r * sin_thetas).astype(int)
            
            # Filter valid coordinates
            valid = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            a_valid = a[valid]
            b_valid = b[valid]
            
            # Increment accumulator
            accumulator[r_idx, b_valid, a_valid] += 1

    # Find peaks in accumulator
    detected_circles = []
    for r_idx, r in enumerate(radii):
        coords = np.argwhere(accumulator[r_idx] > threshold)
        for b, a in coords:
            votes = accumulator[r_idx, b, a]
            detected_circles.append((a, b, r, votes))

    # Sort by votes
    detected_circles.sort(key=lambda x: -x[3])

    # Minimum distance between the centers of the detected circles should be larger than minDist
    if min_dist:
        circles_filtered = []
        for x, y, r, v in detected_circles:
            if all([(x - x1)**2 + (y - y1)**2 > min_dist**2 for x1, y1, _, _ in circles_filtered]):
                circles_filtered.append((x, y, r, v))

        detected_circles = circles_filtered

    return detected_circles, accumulator


def gaussian_kernel(distance, bandwidth):
    """
    Compute Gaussian kernel value.
    
    Args:
        distance: Euclidean distance
        bandwidth: Bandwidth parameter h
        
    Returns:
        Kernel value
    """
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def mean_shift_step(point, data, bandwidth, kernel_func=gaussian_kernel):
    """
    Perform one mean shift iteration step.
    
    Args:
        point: Current point position (numpy array)
        data: All data points (numpy array, shape: [n_points, n_dims])
        bandwidth: Bandwidth parameter (window radius)
        kernel_func: Kernel function to use
        
    Returns:
        new_point: Updated point position
        shift: Magnitude of shift
    """
    # Compute distances from current point to all data points
    distances = np.linalg.norm(data - point, axis=1)
    
    # Only consider points within the bandwidth window
    within_window = distances <= bandwidth
    
    # Compute kernel weights only for points within window
    weights = np.zeros(len(distances))
    weights[within_window] = kernel_func(distances[within_window], bandwidth)
    
    # Avoid division by zero
    total_weight = np.sum(weights)
    if total_weight < 1e-10:
        return point, 0.0
    
    # Compute weighted mean
    new_point = np.sum(weights[:, np.newaxis] * data, axis=0) / total_weight
    
    # Compute shift magnitude
    shift = np.linalg.norm(new_point - point)
    
    return new_point, shift

def mean_shift_converge(initial_point, data, bandwidth, max_iter=100, tol=1e-3):
    """
    Run mean shift until convergence.
    
    Args:
        initial_point: Starting point
        data: All data points
        bandwidth: Bandwidth parameter
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (minimum shift)
        
    Returns:
        converged_point: Final converged position
        trajectory: List of positions during convergence
    """
    trajectory = [initial_point.copy()]
    current_point = initial_point.copy()
    
    for iteration in range(max_iter):
        new_point, shift = mean_shift_step(current_point, data, bandwidth)
        trajectory.append(new_point.copy())
        
        if shift < tol:
            break
        
        current_point = new_point
    
    return current_point, trajectory

def myMeanShift(accumulator, bandwidth=2.0, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift
        threshold: Minimum value to consider (if None, use fraction of max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape
    
    # Set threshold
    if threshold is None:
        threshold = 0.4 * np.max(accumulator)
    
    # Find all points above threshold
    candidate_points = []
    for r_idx in range(n_r):
        y_coords, x_coords = np.where(accumulator[r_idx, :, :] >= threshold)
        for y, x in zip(y_coords, x_coords):
            value = accumulator[r_idx, y, x]
            candidate_points.append([int(x), int(y), int(r_idx), int(value)])
    
    if len(candidate_points) == 0:
        return []
    
    candidate_points = np.array(candidate_points)
    
    # Create feature space
    features = candidate_points[:, :3].copy().astype(float) # (n, 4) -> (n, 3)
    features[:, 0] = features[:, 0] / w  # Normalize x
    features[:, 1] = features[:, 1] / h  # Normalize y
    features[:, 2] = features[:, 2] / n_r  # Normalize r
    
    n_samples = min(200, len(features))
    sample_indices = np.random.choice(len(features), n_samples, replace=False)
    sampled_features = features[sample_indices]
    
    # Run mean shift from each sampled point
    converged_points = []
    for feature in sampled_features:
        converged, _ = mean_shift_converge(feature, features, bandwidth, max_iter=50)
        converged_points.append(converged)
    
    converged_points = np.array(converged_points)

    # Filtering peaks
    peaks = []
    used = np.zeros(len(converged_points), dtype=bool)
    
    for i, point in enumerate(converged_points):
        if used[i]:
            continue
        
        # Find all points close to this one
        distances = np.linalg.norm(converged_points - point, axis=1)
        cluster = distances < 0.2
        used[cluster] = True
        
        # Find original point with highest value in this cluster
        cluster_indices = sample_indices[cluster]
        best_idx = cluster_indices[np.argmax(candidate_points[cluster_indices, 3])]
        
        x, y, r_idx, value = candidate_points[best_idx]
        peaks.append((int(x), int(y), int(r_idx), value))
    
    # Sort by value
    peaks.sort(key=lambda p: p[3], reverse=True)
    
    return peaks



def visualize_detected_circles(img, circles):
    """
    Visualize detected circles on the original image.
    
    Args:
        img: Original image
        circles: List of (x, y, r, votes) tuples
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for x, y, r, votes in circles:
        img = cv2.circle(img, (x,y), 1, (0,255,0), 3)  # center
        img = cv2.circle(img, (x,y), r, (0,0,255), 3)  # circle
    
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Detected Circles: {len(circles)} circles found')
    plt.tight_layout()
    
    return fig

def visualize_accumulator(accumulator, radii, n_slices=4):
    """
    Visualize slices of the 3D accumulator for different radii.
    
    Args:
        accumulator: 3D accumulator array (h, w, n_radii)
        radii: Array of radius values
        n_slices: Number of slices to show
    """
    n_radii = accumulator.shape[0]
    indices = np.linspace(0, n_radii - 1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(1, n_slices, figsize=(16, 4))
    
    for idx, ax in zip(indices, axes):
        im = ax.imshow(accumulator[idx, :, :], cmap='hot')
        ax.set_title(f'Radius = {radii[idx]:.1f}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Accumulator Slices for Different Radii')
    plt.tight_layout()
    
    return fig

def visualize_peak_radius(accumulator, radii, peak_idx=None):
    """
    Visualize the accumulator slice at the radius with maximum votes.
    
    Args:
        accumulator: 3D array of shape (num_radii, height, width)
        radii: 1D array of radius values
        peak_idx: Specific radius index to visualize. If None, uses radius with max votes
    """
    if peak_idx is None:
        peak_idx = np.argmax(np.max(accumulator, axis=(1, 2)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(accumulator[peak_idx, :, :], cmap='hot', interpolation='nearest')
    ax.set_title(f'Accumulator at Peak Radius = {radii[peak_idx]:.1f} pixels', fontsize=14, pad=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Vote Count', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig


def main():
    print("Hough Transform for Circle Detection")

    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\nLoaded image: {img_path}")
    print(f"Image size: {img_gray.shape}")
    
    # Apply Canny edge detection
    print("\nApplying Canny edge detection...")
    img_gray = cv2.medianBlur(img_gray, 5)
    edges = cv2.Canny(img_gray, 50, 150)
    
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    min_radius = 30
    max_radius = 100
    threshold = 40
    min_dist = img.shape[0]/8
    r_ssz = 1
    theta_ssz = 2
    
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    print(f"\nDetected {len(detected_circles)} circles")
    for i, (x, y, r, votes) in enumerate(detected_circles):
        print(f"  Circle {i+1}: center=({x:.1f}, {y:.1f}), radius={r:.1f}, votes={votes:.0f}")
    
    # Visualize detected circles
    os.makedirs('output/q2', exist_ok=True)
    save_path = 'output/q2/detected_circles_{}.png'.format(img_path.split('/')[-1].split('.')[0])
    fig = visualize_detected_circles(img, detected_circles)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Visualize accumulator slices
    save_path2 = 'output/q2/accumulator_slices_{}.png'.format(img_path.split('/')[-1].split('.')[0])
    radii = np.arange(min_radius, max_radius, r_ssz)
    fig = visualize_accumulator(accumulator, radii)
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Visualize peak radius
    save_path2 = 'output/q2/peak_radius_accumulator_{}.png'.format(img_path.split('/')[-1].split('.')[0])
    fig = visualize_peak_radius(accumulator, radii)
    fig.savefig(save_path2, dpi=150, bbox_inches='tight')
    plt.close(fig)


    print("=" * 70)
    print("Mean Shift for Peak Detection in Hough Accumulator")

    print("Applying mean shift to find peaks...")
    # Find peaks using mean shift
    peaks = myMeanShift(accumulator, bandwidth=0.05, threshold=0.4*np.max(accumulator))
    
    print(f"\nDetected {len(peaks)} peaks using mean shift")
    for i, (x, y, r_idx, value) in enumerate(peaks):
        print(f"  Peak {i+1}: center=({x:.1f}, {y:.1f}), radius={radii[r_idx]:.1f}, value={value:.1f}")
        peaks[i] = (x, y, radii[r_idx], value) # update the actual radius for visualization

    # Visualize corresponding circles on original image    
    os.makedirs('output/q3', exist_ok=True)
    fig = visualize_detected_circles(img_copy, peaks)
    fig.savefig('output/q3/circles_from_meanshift.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()