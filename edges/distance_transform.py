import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

'''
This code implements a distance transform using the Chamfer 5-7-11 method. 
It loads an image, applies Canny edge detection to find edges, and then computes the distance transform using both the custom Chamfer method and OpenCV's built-in distance transform.
'''

def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
        
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are 255, background is 0
    
    Returns:
        Distance transform image
    """
    h, w = binary_image.shape
    dt = np.full((h, w), np.inf, dtype=np.float32)
    
    dt[binary_image > 0] = 0
    
    # Define forward and backward masks with (row_offset, col_offset, distance)
    # Forward mask
    forward_offsets = [
        (-2, -1, 11), (-2, 1, 11),  # Two rows up
        (-1, -2, 11), (-1, -1, 7), (-1, 0, 5), (-1, 1, 7), (-1, 2, 11),  # One row up
        (0, -1, 5)  # Same row, left
    ]
    
    # Backward mask
    backward_offsets = [
        (0, 1, 5), # Same row, right
        (1, -2, 11), (1, -1, 7), (1, 0, 5), (1, 1, 7), (1, 2, 11),  # One row down
        (2, -1, 11), (2, 1, 11)  # Two rows down
    ]
    
    # Forward pass
    for i in range(h):
        for j in range(w):
            if dt[i, j] != 0:
                for di, dj, weight in forward_offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        dt[i, j] = min(dt[i, j], dt[ni, nj] + weight)
    
    # Backward pass
    for i in range(h-1, -1, -1):
        for j in range(w-1, -1, -1):
            if dt[i, j] != 0:
                for di, dj, weight in backward_offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        dt[i, j] = min(dt[i, j], dt[ni, nj] + weight)
    
    return dt


def main():    
    print("=" * 70)
    print("Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    img_path = 'data/bonn_distance_transform.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Loaded image: {img_path}")
    print(f"Image size: {gray.shape}")
    
    # Apply Canny edge detection
    print("Applying Canny edge detection...")
    edges = cv2.Canny(gray, 50, 150)
    
    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    print("Computing distance transform...")
    dist_map = chamfer_distance_transform_5_7_11(edges)
    
    # Compute distance transform using cv2.distanceTransform
    euclidean_dist = cv2.distanceTransform(
        (edges == 0).astype(np.uint8), 
        cv2.DIST_L2, 
        cv2.DIST_MASK_PRECISE
    )

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    
    # Edge image
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Canny Edges')
    
    # Distance transform
    im = axes[1, 0].imshow(dist_map, cmap='hot')
    axes[1, 0].set_title('Distance Transform (Chamfer 5-7-11)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Distance transform using OpenCV
    im2 = axes[1, 1].imshow(euclidean_dist, cmap='hot')
    axes[1, 1].set_title('Distance Transform (OpenCV L2)')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.suptitle('Distance Transform Results', fontsize=14)
    plt.tight_layout()
    
    os.makedirs('output/', exist_ok=True)
    save_path = 'output/distance_transform_{}.png'.format(img_path.split('/')[-1].split('.')[0])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
        
    print("\n" + "=" * 70)
    print("Results saved in ", save_path)
    print("=" * 70)


if __name__ == "__main__":
    main()