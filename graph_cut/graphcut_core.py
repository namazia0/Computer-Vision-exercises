import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


'''
This code implements an offline graph cut segmentation algorithm. It takes an input image and user-provided scribbles to separate the foreground from the background. 
The algorithm formulates the problem as energy minimization on a Markov Random Field (MRF) and uses the maxflow library to compute the optimal segmentation. 
The code includes two methods for computing unary potentials: GMM-based and histogram-based. 
Finally, it evaluates the results using Intersection over Union (IoU) against ground truth masks.
'''


class OfflineGraphCut:
    def __init__(self, img_path, label_path, gt_path=None):
        """
        img_path: Path to the input RGB image.
        label_path: Path to stroke image.
        """
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        self.labels_img = cv2.imread(label_path)
        if self.labels_img is None:
            raise FileNotFoundError(f"Labels not found: {label_path}")
        self.gt = cv2.imread(gt_path, 0) if gt_path else None
        self.h, self.w = self.img.shape[:2]
        self.lambda_smoothness = 1000.0 

    def parse_labels(self):
        """
        Scribble interpretation (OpenCV BGR):
        - Near-white strokes => Foreground (mask=1)
        - Red strokes        => Background (mask=2)
        """
        # load scribble image
        lab = self.labels_img
        B = lab[:,:,0]; G = lab[:,:,1]; R = lab[:,:,2]

        mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # Red background strokes (High Red, Low Green/Blue)
        bg_red = (R > 170) & (G < 80) & (B < 80)

        # Near-white foreground strokes (High R, G, B)
        fg_white = (R > 210) & (G > 210) & (B > 180)

        # 1: Foreground, 2: Background
        mask[fg_white] = 1
        mask[bg_red]   = 2

        return mask
    


    # The Unary Potential measures how likely a pixel belongs to FG or BG based solely on its color. 
    # We implement two probabilistic models. The cost is the negative log-likelihood.

    def compute_unary_gmm(self, mask):
        
        """
        Method 1: GMM-based Unary Potentials 
        """
        img_flat = self.img.reshape(-1, 3)
        
        fg_pixels = self.img[mask == 1]
        bg_pixels = self.img[mask == 2]

        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            raise ValueError("Foreground or Background scribbles missing.")
    
        # Keep components low to prevent overfitting
        n_comp_fg = min(5, len(fg_pixels))
        n_comp_bg = min(5, len(bg_pixels))
        
        # Use 'full' covariance to capture RGB correlations 
        gmm_fg = GaussianMixture(n_components=n_comp_fg, covariance_type='full', reg_covar=1e-6, random_state=1142)
        # We learn by fitting the GMM to the pixel colors in the scribbled FG region:
        # Mean color: "What is the average foreground color?"
        # Covariance: "How much does the foreground color vary?"
        gmm_fg.fit(fg_pixels)
        
        # Same for Background
        gmm_bg = GaussianMixture(n_components=n_comp_bg, covariance_type='full', reg_covar=1e-6, random_state=1142)
        gmm_bg.fit(bg_pixels)

        # We take the fitted GMMs and evaluate the log-likelihood of each pixel in the entire image:
        log_p_fg = gmm_fg.score_samples(img_flat).reshape(self.h, self.w)
        log_p_bg = gmm_bg.score_samples(img_flat).reshape(self.h, self.w)

        # Convert log-likelihoods to costs
        # Graph Cut requires non-negative costs, where 0 is the best possible match.
        D_fg = -(log_p_fg - np.max(log_p_fg))
        D_bg = -(log_p_bg - np.max(log_p_bg))

        return D_fg, D_bg


    def compute_unary_hist(self, mask):
        """
        Method 2: Histogram-based Unary Potentials 
        """
        fg_mask = (mask == 1).astype(np.uint8) * 255
        bg_mask = (mask == 2).astype(np.uint8) * 255

        if fg_mask.sum() == 0 or bg_mask.sum() == 0:
            raise ValueError("Foreground or Background scribbles missing.")

        bins = 32
        ranges = [0, 256, 0, 256, 0, 256]
        
        # Calculate Histograms: a look-up table of color frequencies
        hist_fg = cv2.calcHist([self.img], [0,1,2], fg_mask, [bins,bins,bins], ranges)
        hist_bg = cv2.calcHist([self.img], [0,1,2], bg_mask, [bins,bins,bins], ranges)
        
        # Normalize (PDF approximation)
        # Normalize counts of FGs and BGs regardless of how many pixels scribbled
        # Most common color 1, rare color 0.00001
        cv2.normalize(hist_fg, hist_fg, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_bg, hist_bg, 0, 1, cv2.NORM_MINMAX)

        # We convert each pixel's RGB value to a histogram bin index
        bin_idxs = (self.img // (256 // bins)).astype(int)

        p_fg = hist_fg[bin_idxs[:,:,0], bin_idxs[:,:,1], bin_idxs[:,:,2]]
        p_bg = hist_bg[bin_idxs[:,:,0], bin_idxs[:,:,1], bin_idxs[:,:,2]]

        # Negative Log Likelihood for conversion to costs
        eps = 1e-10
        D_fg = -np.log(p_fg + eps)
        D_bg = -np.log(p_bg + eps)

        return D_fg, D_bg
    

    def compute_pairwise_potentials(self):
        """
        Computes the smoothness term (Pairwise Potentials) based on image contrast.
        Returns horizontal (w_right) and vertical (w_down) edge weights.
        Shared by both GMM and Histogram methods.
        """
        img_float = self.img.astype(np.float32)
        # vector differences
        diff_right = img_float[:, :-1] - img_float[:, 1:]
        diff_down  = img_float[:-1, :] - img_float[1:, :]

        sq_diff_sum = np.sum(diff_right**2) + np.sum(diff_down**2)
        num_edges = diff_right.size + diff_down.size
        
        if sq_diff_sum < 1e-5:
            beta = 0
        else:
            beta = 1.0 / (2.0 * (sq_diff_sum / num_edges))

        # Calculate edge weights
        sq_diff_right = np.sum(diff_right**2, axis=2)
        sq_diff_down  = np.sum(diff_down**2, axis=2)
        
        w_right = self.lambda_smoothness * np.exp(-beta * sq_diff_right)
        w_down  = self.lambda_smoothness * np.exp(-beta * sq_diff_down)

        return w_right, w_down



    
    def build_graph(self, mode='gmm'):
        mask = self.parse_labels()
        
        # Select Unary Computation Method
        # Cost of calling a pixel Foreground and Background.
        if mode == 'gmm':
            D_fg, D_bg = self.compute_unary_gmm(mask)
        elif mode == 'hist':
            D_fg, D_bg = self.compute_unary_hist(mask)
        else:
            raise ValueError("Mode must be 'gmm' or 'hist'")

        INF = 1e9
        D_fg[mask == 1] = 0; D_bg[mask == 1] = INF
        D_fg[mask == 2] = INF; D_bg[mask == 2] = 0

        # Initialize Graph
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((self.h, self.w))
        
        # Add T-Edges (Data Term)
        # add_grid_tedges(ids, capacity_to_source, capacity_to_sink)
        g.add_grid_tedges(nodeids, 
                          np.ascontiguousarray(D_bg, dtype=np.float64), 
                          np.ascontiguousarray(D_fg, dtype=np.float64))

        # Add N-Edges (Smoothness Term)
        w_right, w_down = self.compute_pairwise_potentials()
        
        # Pad weights to match grid size
        w_right_full = np.zeros((self.h, self.w), dtype=np.float64)
        w_right_full[:, :-1] = w_right
        w_down_full = np.zeros((self.h, self.w), dtype=np.float64)
        w_down_full[:-1, :] = w_down

        struct_right = np.array([[0, 0, 0], 
                                 [0, 0, 1], 
                                 [0, 0, 0]], dtype=np.float64)
        
        struct_down  = np.array([[0, 0, 0], 
                                 [0, 0, 0], 
                                 [0, 1, 0]], dtype=np.float64)

        g.add_grid_edges(nodeids, np.ascontiguousarray(w_right_full), struct_right, symmetric=True)
        g.add_grid_edges(nodeids, np.ascontiguousarray(w_down_full), struct_down, symmetric=True)
        
        return g, nodeids
    

    def run(self, mode='gmm'):
        # Initialize Graph
        g, nodeids = self.build_graph(mode=mode)
        
        # Max Flow Calculation
        flow = g.maxflow()
        
        # Get result (False=Source/FG, True=Sink/BG)
        sgm = g.get_grid_segments(nodeids)
        
        # Convert to binary mask (BG(True)=0, FG(False)=255)
        self.pred_mask = np.where(sgm, 0, 255).astype(np.uint8)
        
        return self.pred_mask

    def evaluate(self):
        if self.gt is None:
            return 0.0
        
        # Calculate Intersection over Union
        intersection = np.logical_and(self.pred_mask == 255, self.gt == 255)
        union = np.logical_or(self.pred_mask == 255, self.gt == 255)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-10)
        return iou


if __name__ == "__main__":
    
    base_dir = "dataset" 

    images_dir = os.path.join(base_dir, "images")
    hist_scores = []
    gmm_scores = []
    # Check if dir exists
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} not found.")
    else:
        file_list = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        
        for file in file_list:
            obj_name = file[:-4]
            img_path = os.path.join(base_dir, "images", f"{obj_name}.jpg")
            lbl_path = os.path.join(base_dir, "images-labels", f"{obj_name}-anno.png")
            gt_path  = os.path.join(base_dir, "images-gt", f"{obj_name}.png")
            
            print(f"Processing: {obj_name}")

            try:
                # Initialize Segmenter
                seg = OfflineGraphCut(img_path, lbl_path, gt_path)
                
                # Run GMM Method
                mask_gmm = seg.run(mode='gmm')
                iou_gmm = seg.evaluate()
                gmm_scores.append(iou_gmm)
                
                # Run Histogram Method
                mask_hist = seg.run(mode='hist')
                iou_hist = seg.evaluate()
                hist_scores.append(iou_hist)
                
                print(f"  > GMM IoU:  {iou_gmm:.4f}")
                print(f"  > Hist IoU: {iou_hist:.4f}")

                # Visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 5, 1); plt.title("Input Image")
                plt.imshow(cv2.cvtColor(seg.img, cv2.COLOR_BGR2RGB)); plt.axis('off')
                
                plt.subplot(1, 5, 2); plt.title("Scribbles\n(Red=BG, White=FG)")
                plt.imshow(cv2.cvtColor(seg.labels_img, cv2.COLOR_BGR2RGB)); plt.axis('off')
                
                plt.subplot(1, 5, 3); plt.title(f"GMM Result\nIoU: {iou_gmm:.3f}")
                plt.imshow(mask_gmm, cmap='gray'); plt.axis('off')
                
                plt.subplot(1, 5, 4); plt.title(f"Hist Result\nIoU: {iou_hist:.3f}")
                plt.imshow(mask_hist, cmap='gray'); plt.axis('off')

                if seg.gt is not None:
                    plt.subplot(1, 5, 5); plt.title("Ground Truth")
                    plt.imshow(seg.gt, cmap='gray'); plt.axis('off')
                
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Skipping {obj_name}: {e}")

    # Final Results
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    if hist_scores:
        avg_hist = sum(hist_scores) / len(hist_scores)
        print(f"Average Histogram IoU: {avg_hist:.4f} (on {len(hist_scores)} images)")
    if gmm_scores:
        avg_gmm = sum(gmm_scores) / len(gmm_scores)
        print(f"Average GMM IoU: {avg_gmm:.4f} (on {len(gmm_scores)} images)")
    print("="*30)