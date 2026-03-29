import numpy as np
import cv2
import random

np.set_printoptions(suppress=True)

'''
This file contains the implementation of panorama stitching (image mosaic) for multiple images using feature-based homography estimation and blending.
It is an automatic multi-image panorama stitcher.
'''

def compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2, nSamples=4, nIterations=20, thresh=0.1):
    # compute best H transformation using RANSAC algorithm
    # RANSAC loop
    best_mse = 1e20
    best_H = None
    for i in range(nIterations):

        print('iteration ' + str(i))

        # randomly select some keypoints
        rand_matches = [match[0] for match in random.sample(good_matches, nSamples)]

        pts1 = [[kp_1[match.queryIdx].pt[0], kp_1[match.queryIdx].pt[1]] for match in rand_matches]
        pts2 = [[kp_2[match.trainIdx].pt[0], kp_2[match.trainIdx].pt[1]] for match in rand_matches]

        hom = cv2.getPerspectiveTransform(np.float32(pts2), np.float32(pts1))
        warpedImg2 = cv2.warpPerspective(image_2, hom, (image_1.shape[1], image_1.shape[0]))

        total_mse = 0
        inliers_count = 0

        for test_kp in [kp for kp in kp_1 if kp.size > 0]:

            size = test_kp.size

            min_r, max_r, min_c, max_c = test_kp.pt[1] - size / 2, test_kp.pt[1] + size / 2, test_kp.pt[0] - size / 2, \
                                         test_kp.pt[0] + size / 2

            patch1 = image_1[int(min_r):int(max_r), int(min_c):int(max_c), :].astype(np.float32)
            patch2 = warpedImg2[int(min_r):int(max_r), int(min_c):int(max_c), :].astype(np.float32)

            diff = patch1 - patch2

            mse = np.sum(np.multiply(diff, diff))

            mse /= (size * size * 3 * 255 * 255)

            if mse < thresh:
                inliers_count += 1
                total_mse += mse

        total_mse /= inliers_count + 1e-12

        if total_mse < best_mse:
            best_H = hom
            best_mse = total_mse

    return best_H


def get_best_match(des_1, des_2, thr=0.3):
    dist = np.expand_dims(des_1, axis=1) - np.expand_dims(des_2, axis=0)  # 1032, 1079, 128
    dist = np.sum(np.multiply(dist, dist), axis=2)
    match12 = np.argmin(dist, axis=1)
    match21 = np.argmin(dist, axis=0)
    distances12 = np.min(dist, axis=1)

    good_matches = []

    for queryIdx, trainIdx in enumerate(match12):
        if match21[trainIdx] == queryIdx and thr * dist[queryIdx, np.argpartition(dist[queryIdx, :], 2)[1]] > \
                distances12[queryIdx]:
            good_matches.append([cv2.DMatch(queryIdx, trainIdx, distances12[queryIdx])])

    return good_matches


def compute_panorama_size(images, homographies):
    """Compute the bounding box for the final panorama given all images and their homographies."""

    all_corners = []

    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix to shift everything to positive coordinates
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]], dtype=np.float64)

    output_size = (x_max - x_min, y_max - y_min)

    return output_size, translation


def stitch_multiple_images(images, homographies):
    """Stitch multiple images given their homographies to a common reference frame."""

    # Compute the size of the final panorama and the translation needed
    output_size, translation = compute_panorama_size(images, homographies)

    # Initialize the result panorama
    result = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Warp and blend each image
    for img, H in zip(images, homographies):
        # Combine translation with homography
        H_translated = translation @ H

        # Warp the image
        warped = cv2.warpPerspective(img, H_translated, output_size)

        # Create mask for this warped image
        mask = (np.sum(warped, axis=2) > 0).astype(np.uint8)

        # Create mask for existing result
        result_mask = (np.sum(result, axis=2) > 0).astype(np.uint8)

        # Find overlapping region
        overlap = (mask > 0) & (result_mask > 0)

        # Find non-overlapping region of new image
        new_only = (mask > 0) & (result_mask == 0)

        # Add non-overlapping regions directly
        result[new_only] = warped[new_only]

        # Blend overlapping regions (simple 50/50 blend)
        if np.any(overlap):
            result[overlap] = (result[overlap].astype(np.float32) * 0.5 +
                               warped[overlap].astype(np.float32) * 0.5).astype(np.uint8)

    return result


def create_panorama():
    # Load all images
    image_paths = [r'./data/Fuji_1.png', r'./data/Fuji_2.png', r'./data/Fuji_3.png']
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        images.append(img)

    sift = cv2.SIFT_create()

    # Extract keypoints and descriptors for all images
    keypoints = []
    descriptors = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    num_imgs = len(images)
    connectivity = np.zeros(num_imgs)

    # Compute matches between all pairs
    for i in range(num_imgs):
        for j in range(i + 1, num_imgs):
            print(f"Processing pair {i} - {j}")
            # Match
            good_matches = get_best_match(descriptors[i], descriptors[j], thr=0.3)

            # Visualize matches
            img_matches = cv2.drawMatchesKnn(images[i], keypoints[i], images[j], keypoints[j], good_matches, None, flags=2)
            cv2.imshow(f'Matches {i}-{j}', img_matches)
            cv2.waitKey(0)
            cv2.destroyWindow(f'Matches {i}-{j}')

            # save the matches visualization
            cv2.imwrite(f'output/sift_matches_{i}_{j}.png', img_matches)

            # Update connectivity
            connectivity[i] += len(good_matches)
            connectivity[j] += len(good_matches)

    cv2.destroyAllWindows()

    # Find optimal reference image based on connectivity
    ref_idx = int(np.argmax(connectivity))
    print(f"Optimal reference image index: {ref_idx} (Connectivity: {connectivity[ref_idx]})")

    # Compute homographies to reference
    homographies = []
    for i in range(num_imgs):
        if i == ref_idx:
            homographies.append(np.eye(3, dtype=np.float64))
        else:
            print(f"Computing homography for image {i} -> reference {ref_idx}...")
            # We warp image i to reference, so Ref is query (destination), i is train (source)
            matches = get_best_match(descriptors[ref_idx], descriptors[i], thr=0.3)
            H = compute_Homography_RANSAC(matches, keypoints[ref_idx], keypoints[i], images[ref_idx], images[i])
            homographies.append(H)

    print("Creating final panorama...")

    final_panorama = stitch_multiple_images(images, homographies)

    # Display final result
    cv2.namedWindow('Final Panorama', cv2.WINDOW_NORMAL)
    cv2.imshow('Final Panorama', final_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the result
    cv2.imwrite('output/final_panorama.png', final_panorama)
    print("Final panorama saved as 'output/final_panorama.png'")


if __name__ == "__main__":
    create_panorama()