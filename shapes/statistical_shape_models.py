
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

'''
This code implements Procrustes Analysis to align a set of hand keypoints and then trains a statistical shape model using PCA and PPCA. 
It visualizes the impact of principal components on the shape variation and reconstructs a test shape using the learned model.
'''

# Load train and test keypoints
kpts = np.load('hands_train.npy')
test_kpts = np.load('hands_test.npy')
print(kpts.shape, test_kpts.shape)


def calculate_mean_shape(kpts):
    mean_shape = np.mean(kpts, axis=0)
    return mean_shape

# Solve for affine transformation from each shape to the mean shape
def affine_transf(kpts, reference_mean):
    b, n = kpts.shape[0], kpts.shape[1]
    sol = reference_mean.flatten()  # 1 x N * 2
  
    # Fill in the matrix
    A = np.zeros((b, n, 2, 6))
    A[..., 0, 0] = kpts[..., 0]
    A[..., 0, 1] = kpts[..., 1]
    A[..., 0, 2] = 1

    A[..., 1, -1] = 1
    A[..., 1, -2] = kpts[..., 1]
    A[..., 1, -3] = kpts[..., 0]
  
    A = np.reshape(A, (b, n * 2, 6))
    pi_A = np.linalg.pinv(A)
    affine_transf = np.dot(pi_A, sol) 

    # get rotation and translation
    rot_scale = np.zeros((b, 2, 2))
    transl = np.zeros((b, 2))

    rot_scale[:, 0, 0] = affine_transf[:, 0]
    rot_scale[:, 0, 1] = affine_transf[:, 1]
    rot_scale[:, 1, 0] = affine_transf[:, 3]
    rot_scale[:, 1, 1] = affine_transf[:, 4]
    
    transl[:, 0] = affine_transf[:, 2]
    transl[:, 1] = affine_transf[:, 5]

    return rot_scale, transl


def procrustres_analysis_step(kpts, reference_mean):
    rot, transl = affine_transf(kpts, reference_mean)
    rotated_kpts = np.asarray([np.dot(rot[i], kpts[i].T).T for i in range(rot.shape[0])])
    transl_kpts = rotated_kpts + transl[:, None, :]
    return transl_kpts


def compute_avg_error(kpts, mean_shape):
    squared_diff = (kpts - mean_shape[None, :, :]) ** 2
    mse = np.mean(squared_diff)
    return mse


def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-2):
    aligned_kpts = kpts.copy()

    # Take random sample as initial mean
    reference_mean = kpts[-1]

    for iter in range(max_iter):
        # align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_mean)

        # calculate new reference mean
        reference_mean = calculate_mean_shape(aligned_kpts)

        # calculate alignment error
        mse = compute_avg_error(aligned_kpts, reference_mean)

        print("(%d) MSE: %f" % (iter + 1, mse))

        if mse <= min_error:
            break

        reference_mean = calculate_mean_shape(aligned_kpts)

    return aligned_kpts, reference_mean


aligned_kpts, reference_mean = procrustres_analysis(kpts)
print(aligned_kpts.shape)

# Plot initial and aligned keypoints in side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].set_title('Original Keypoints')
axs[0].invert_yaxis()
axs[0].axis('off')
for i in range(kpts.shape[0]):
    axs[0].plot(kpts[i, :, 0], kpts[i, :, 1], marker='o', linestyle='-')
axs[1].set_title('Aligned Keypoints')
axs[1].invert_yaxis()
axs[1].axis('off')
for i in range(aligned_kpts.shape[0]):
    axs[1].plot(aligned_kpts[i, :, 0], aligned_kpts[i, :, 1], marker='o', linestyle='-')

plt.savefig('output/hands.png')


def ppca(covariance):
    U, s, VT = np.linalg.svd(covariance, full_matrices=True)
    eigenvectors = U
    eigenvalues = s

    idxs = np.argsort(s)[::-1]
    eigenvectors = U[idxs, :]
    eigenvalues = s[idxs]

    eigval_sum = np.sum(eigenvalues)
    cum = 0
    for i in range(len(eigenvalues)):
        cum += eigenvalues[i]
        if cum / eigval_sum >= 0.9:
            break

    print("We select %d PCs" % (i + 1))

    selected_eigenvalues = eigenvalues[:(i+1)]
    selected_pcs = eigenvectors[:, :(i+1)]
    selected_pcs = selected_pcs / np.sqrt(selected_eigenvalues[None, :]) 
   
    # Probabilistic version
    sigma = np.sum(eigenvalues[(i+1):]) / (len(eigenvalues) - (i + 1)) 
    noise = (eigenvalues[:(i+1)] - sigma) ** 0.5 
    selected_pcs = selected_pcs @ np.diag(noise)
    return selected_pcs.T, selected_eigenvalues, sigma


def create_covariance_matrix(kpts, mean_shape):
    aligned_kpts = kpts - mean_shape[np.newaxis, :]
    covariance = np.dot(aligned_kpts.transpose(), aligned_kpts)
    return covariance


def visualize_impact_of_pcs(mean, pcs, pc_weights, probabilistic):
    weights = np.linspace(-0.3, 0.3, 7) 
    for pc_idx in range(len(pc_weights)):
        plt.figure(figsize=(4, 4))
        plt.axis("off")
        plt.gca().invert_yaxis()
        for w in weights:
            shape = mean + w * np.sqrt(pc_weights[pc_idx]) * pcs[pc_idx]
            shape = shape.reshape(mean.shape[0] // 2, 2)

            plt.plot(shape[:, 0], shape[:, 1], linewidth=1)

        plt.title(f"PC {pc_idx}")
        plt.savefig(f"output/var_{pc_idx}{'_prob' if probabilistic else ''}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
                

def train_statistical_shape_model(kpts, probabilistic=False):

    # Mean Shape
    mean_shape = np.mean(kpts, axis=0)

    # COVAR
    covariance = create_covariance_matrix(kpts, mean_shape)

    # PCA
    if probabilistic:
        pcs, pc_weights, sigma = ppca(covariance)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idxs]
        eigenvalues = eigenvalues[idxs]

        eigval_sum = np.sum(eigenvalues)
        cum = 0
        for i in range(len(eigenvalues)):
            cum += eigenvalues[i]
            if cum / eigval_sum >= 0.9:
                break

        print("We select %d PCs" % (i + 1))

        pc_weights = eigenvalues[:(i+1)]
        pcs = np.moveaxis(eigenvectors[:, :(i+1)], 1, 0)

    # VIS
    visualize_impact_of_pcs(mean_shape, pcs, pc_weights, probabilistic)

    return mean_shape, pcs, pc_weights, sigma if probabilistic else 0


def reconstruct_test_shape(orig_kpts, kpts, mean, pcs):
    print(kpts.shape)

    # Align test shape
    aligned_kpts = kpts - mean[None, :]
    c = np.dot(pcs, aligned_kpts[0])
    weighted_pcs = c[:, None] * pcs 
    reconstruction = mean + np.sum(weighted_pcs, axis=0) #+ np.random.normal(scale=np.sqrt(sigma), size=kpts.shape)

    # calculate reconstruction error
    error = np.sqrt(np.mean((kpts - reconstruction) ** 2))
    print("RMS:", error)

    # Visualize original, aligned, and reconstructed hand side-by-side
    kpts = kpts.reshape(-1, 2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title('Original Keypoints')
    axs[0].invert_yaxis()
    axs[0].axis('off')
    axs[0].plot(orig_kpts[:, 0], orig_kpts[:, 1], marker='o', linestyle='-')
    axs[1].set_title('Aligned Keypoints')
    axs[1].invert_yaxis()
    axs[1].axis('off')
    axs[1].plot(kpts[:, 0], kpts[:, 1], marker='o', linestyle='-')
    axs[2].set_title('Reconstructed Keypoints')
    axs[2].invert_yaxis()
    axs[2].axis('off')
    axs[2].plot(reconstruction.reshape(-1, 2)[:, 0], reconstruction.reshape(-1, 2)[:, 1], marker='o', linestyle='-')
    plt.savefig('output/reconstruction.png')

    return reconstruction


flat_kpts = aligned_kpts.reshape([aligned_kpts.shape[0], -1])
print(aligned_kpts.shape, flat_kpts.shape, test_kpts[None].shape)
mean, pcs, pc_weights, sigma = train_statistical_shape_model(flat_kpts, probabilistic=True)
mean, pcs, pc_weights, sigma = train_statistical_shape_model(flat_kpts)

# Procrustes alignment of test_kpts
aligned_test_kpts = procrustres_analysis_step(test_kpts[None], reference_mean)
reconstruction = reconstruct_test_shape(test_kpts, aligned_test_kpts.flatten(), mean, pcs)

# Print MSE
print(aligned_test_kpts.shape, reconstruction.shape)
mse = np.sum((aligned_test_kpts.flatten() - reconstruction) ** 2) / aligned_test_kpts.flatten().shape[0]