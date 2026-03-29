import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

'''
This code implements a Mixture of Gaussians (MOG) background subtraction algorithm.
'''

class MOG():
    def __init__(self,height=None, width=None, number_of_gaussians=None, background_thresh=None, lr=None):
        self.number_of_gaussians = number_of_gaussians
        self.background_thresh = background_thresh
        self.dist_thresh = 20
        self.lr = lr
        self.height = height
        self.width = width
        self.mus = np.zeros((self.height, self.width, self.number_of_gaussians, 3)) # assuming using color frames
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians)) # all color channels share the same sigma and covariance matrices are diagnalized
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians))
        for i in range(self.height):
            for j in range(self.width):
                self.mus[i,j]=np.array([[122, 122, 122]]*self.number_of_gaussians) # assuming a [0,255] color channel
                self.sigmaSQs[i,j]=[36.0] * self.number_of_gaussians
                self.omegas[i,j]=[1.0 / self.number_of_gaussians] * self.number_of_gaussians

    # used the background adaptation method from lecture 8 (slide 48)
    def updateParam(self, img, BG_pivot): #finish this function
        h, w, c = img.shape
        for i in range(h):
            for j in range(w):
                x = img[i, j]
                mu = self.mus[i, j]
                sigmaSQ = self.sigmaSQs[i, j]
                std = np.sqrt(sigmaSQ)
                omega = self.omegas[i, j]

                distribution_thresh = self.dist_thresh * std

                # Compute the distances between the pixel and each Gaussian mean
                distances = np.linalg.norm(x - mu, axis=1)
            
                M = distances < distribution_thresh
                # Check if there is a match, if yes, update the parameters
                if np.any(M):
                    k = np.argmin(distances)
                    # Update weights
                    M = np.zeros(self.number_of_gaussians)
                    M[k] = 1
                    omega = (1 - self.lr) * omega + self.lr * M
                    # Update the mean
                    rho = self.lr * multivariate_normal.pdf(x, mean=mu[k], cov=np.eye(3) * sigmaSQ[k])
                    mu[k] = (1 - rho) * mu[k] + rho * x
                    # Update the covariance
                    sigmaSQ[k] = (1 - rho) * sigmaSQ[k] + rho * (x - mu[k]) @ (x - mu[k])
                # no match: create new distribution
                else:
                    # choose the least probable distribution to replace
                    k = np.argmin(omega)
                    omega[k] = 0.01
                    mu[k] = x
                    sigmaSQ[k] = 36.0

                # normalize weights
                omega = omega / np.sum(omega)

                # determine the background model
                sorting_index = np.argsort(omega / std)

                sum = 0
                background_model = []
                for idx in reversed(sorting_index):
                    sum += omega[idx]
                    background_model.append(idx.item())
                    if sum > self.background_thresh:
                        break

                # determine if the pixel is background or foreground
                for component in background_model:
                    distance = np.linalg.norm(x - mu[component])
                    if distance < self.dist_thresh * np.sqrt(sigmaSQ[component]):
                        BG_pivot[i, j] = 0
                        break

                # save updated parameters
                self.omegas[i, j] = omega
                self.mus[i, j] = mu
                self.sigmaSQs[i, j] = sigmaSQ

        return (BG_pivot * 255).astype(int)
    
if __name__ == '__main__':
    for i in range(1, 3+1): # display first 3 labeled foreground images
        img = cv2.imread('imgs/{:04d}.jpg'.format(i))

        h, w, c = img.shape
        mog=MOG(height=h, width=w, number_of_gaussians=3, background_thresh=0.5, lr=0.01)

        label_img = mog.updateParam(img, np.ones(img.shape[:2]))

        plt.imshow(label_img, cmap='gray')
        plt.axis("off")
        plt.show()

        cv2.imwrite('output/label{:04d}.jpg'.format(i), label_img)
        print("Processed frame {:04d}".format(i))