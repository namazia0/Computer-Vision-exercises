import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import filedialog


'''
This code implements an interactive graph cut segmentation tool. Users can load an image, draw foreground and background scribbles, and run graph cut segmentation to separate the object from the background.
The tool supports two methods for computing unary potentials: GMM-based and histogram-based. 
'''


class InteractiveGraphCut:
    def __init__(self, img, gt_img=None):
        self.img = img
        self.gt = gt_img
        self.h, self.w = self.img.shape[:2]
        self.lambda_smoothness = 1000.0  
        self.mode = 'gmm'  # Default mode



    def compute_unary_gmm(self, mask):
        
        """
        Method 1: GMM-based Unary Potentials 
        """
        img_flat = self.img.reshape(-1, 3)
        
        fg_pixels = self.img[mask == 1]
        bg_pixels = self.img[mask == 2]

        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            raise ValueError("Foreground or Background scribbles are missing.")

    
        # Keep components low to prevent overfitting
        n_comp_fg = min(5, len(fg_pixels))
        n_comp_bg = min(5, len(bg_pixels))
        
        # Use 'full' covariance to capture RGB correlations 
        gmm_fg = GaussianMixture(n_components=n_comp_fg, covariance_type='full', reg_covar=1e-6, random_state=1142)
        gmm_fg.fit(fg_pixels)
        
        gmm_bg = GaussianMixture(n_components=n_comp_bg, covariance_type='full', reg_covar=1e-6, random_state=1142)
        gmm_bg.fit(bg_pixels)

        log_p_fg = gmm_fg.score_samples(img_flat).reshape(self.h, self.w)
        log_p_bg = gmm_bg.score_samples(img_flat).reshape(self.h, self.w)

        # Normalize costs: Best pixel gets 0 cost
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
            raise ValueError("Foreground or Background scribbles are missing.")

        bins = 32
        ranges = [0, 256, 0, 256, 0, 256]
        
        # Calculate Histograms
        hist_fg = cv2.calcHist([self.img], [0,1,2], fg_mask, [bins,bins,bins], ranges)
        hist_bg = cv2.calcHist([self.img], [0,1,2], bg_mask, [bins,bins,bins], ranges)
        
        # Normalize (PDF approximation)
        cv2.normalize(hist_fg, hist_fg, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_bg, hist_bg, 0, 1, cv2.NORM_MINMAX)

        # Back-project histogram probabilities onto image
        bin_idxs = (self.img // (256 // bins)).astype(int)
        # Clip indices to safe range just in case
        bin_idxs = np.clip(bin_idxs, 0, bins-1)
        
        p_fg = hist_fg[bin_idxs[:,:,0], bin_idxs[:,:,1], bin_idxs[:,:,2]]
        p_bg = hist_bg[bin_idxs[:,:,0], bin_idxs[:,:,1], bin_idxs[:,:,2]]

        # Negative Log Likelihood
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

        # Calculate Beta (expectation of contrast)
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



    
    def build_graph(self, mask_scribbles):
  
        # Select Unary Computation Method
        if self.mode == 'gmm':
            D_fg, D_bg = self.compute_unary_gmm(mask_scribbles)
        elif self.mode == 'hist':
            D_fg, D_bg = self.compute_unary_hist(mask_scribbles)
        else:
            raise ValueError("Mode must be 'gmm' or 'hist'")

        # Apply Hard Constraints (Seeds)
        INF = 1e9
        D_fg[mask_scribbles == 1] = 0; D_bg[mask_scribbles == 1] = INF
        D_fg[mask_scribbles == 2] = INF; D_bg[mask_scribbles == 2] = 0

        # Initialize Graph
        g = maxflow.Graph[float]()
        nodeids = g.add_grid_nodes((self.h, self.w))
        
        # Add T-Edges (Data Term)
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
    

    def run(self, mask_scribbles):
        g, nodeids = self.build_graph(mask_scribbles)
        
        # Max Flow Calculation
        flow = g.maxflow()
        
        # Get result (False=Source/FG, True=Sink/BG)
        sgm = g.get_grid_segments(nodeids)
        
        # Convert to binary mask (0 or 255)
        self.pred_mask = np.where(sgm, 0, 255).astype(np.uint8)
        
        return self.pred_mask

    def evaluate(self, pred_mask):
        if self.gt is None:
            return 0.0
        
        # Calculate Intersection over Union (IoU)
        intersection = np.logical_and(pred_mask == 255, self.gt == 255)
        union = np.logical_or(pred_mask == 255, self.gt == 255)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-10)
        return iou



class InteractiveSegmenter:
    def __init__(self):
        self.window_name = "Interactive Graph Cut (f:FG, b:BG, Space:Run, R:Reset, S:Save)"
        self.brush_size = 5
        self.drawing = False
        self.mode = 0 # 0=Idle, 1=FG (White), 2=BG (Red)
        self.dirty = False # Flag if scribbles changed
        
        # Load Image
        self.load_image_dialog()
        if self.img is None: return

        # Scribble Mask: 0=None, 1=FG, 2=BG
        self.scribble_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        
        # Display Buffer
        self.display_img = self.img.copy()
        self.result_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.overlay_alpha = 0.4

        # Graph Cut Model
        self.model = InteractiveGraphCut(self.img, self.gt)

        # Setup Window
        cv2.namedWindow(self.window_name)
        # Setup Mouse Event Handling
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Controls:")
        print("  [f] : Foreground Brush (White)")
        print("  [b] : Background Brush (Red)")
        print("  [SPACE] : Run Segmentation")
        print("  [m] : Toggle Mode (GMM/Hist)")
        print("  [r] : Reset Scribbles")
        print("  [s] : Save Result")
        print("  [ESC] : Quit")
        
        self.run_loop()

    def load_image_dialog(self):
        root = tk.Tk()
        root.withdraw() # Hide small tkinter window
        file_path = filedialog.askopenfilename(
            title="Select Image for Segmentation",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not file_path:
            print("No file selected.")
            self.img = None
            return

        self.img_path = file_path
        self.img = cv2.imread(file_path)
        
        base_dir = os.path.dirname(os.path.dirname(file_path))
        filename = os.path.splitext(os.path.basename(file_path))[0]
        gt_path = os.path.join(base_dir, "images-gt", f"{filename}.png")
        
        if os.path.exists(gt_path):
            self.gt = cv2.imread(gt_path, 0)
            print(f"Loaded GT: {gt_path}")
        else:
            self.gt = None
            print("No Ground Truth found")

    # Capture Mouse Events
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            if self.mode != 0:
                self.draw_circle(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.mode != 0:
                self.draw_circle(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    # Draw circle on mask and display
    def draw_circle(self, x, y):
        # Update logical mask
        cv2.circle(self.scribble_mask, (x, y), self.brush_size, self.mode, -1)
        # Update visualization immediately
        color = (255, 255, 255) if self.mode == 1 else (0, 0, 255)
        cv2.circle(self.display_img, (x, y), self.brush_size, color, -1)
        self.dirty = True

    def update_visualization(self):
        # Start with clean image
        vis = self.img.copy()
        
        # Draw Scribbles
        # FG (White)
        vis[self.scribble_mask == 1] = [255, 255, 255]
        # BG (Red)
        vis[self.scribble_mask == 2] = [0, 0, 255]

        # Overlay Segmentation Result (Green Tint)
        if np.sum(self.result_mask) > 0:
            green_mask = np.zeros_like(vis)
            green_mask[:, :] = [0, 255, 0] # Green
            # Apply only where mask is FG
            fg_locs = self.result_mask == 255
            
            # Alpha Blending (Transparency)
            vis[fg_locs] = cv2.addWeighted(vis[fg_locs], 1 - self.overlay_alpha, 
                                           green_mask[fg_locs], self.overlay_alpha, 0)

        self.display_img = vis

    def run_segmentation(self):
        print("Running Graph Cut...", end="", flush=True)
        try:
            self.result_mask = self.model.run(self.scribble_mask)
            self.update_visualization()
            print(" Done.")
            
            if self.model.gt is not None:
                iou = self.model.evaluate(self.result_mask)
                print(f"Current IoU: {iou:.4f}")
                # Overlay IoU text on image
                cv2.putText(self.display_img, f"IoU: {iou:.4f} ({self.model.mode})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(self.display_img, f"Mode: {self.model.mode}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
        except Exception as e:
            print(f"\nError: {e}")

    def save_result(self):
        filename = os.path.basename(self.img_path)
        name, _ = os.path.splitext(filename)
        out_path = f"output/{name}_mask_best.png"
        cv2.imwrite(out_path, self.result_mask)
        print(f"Saved binary mask to: {out_path}")

    # The Interaction Loop
    def run_loop(self):
        while True:
            cv2.imshow(self.window_name, self.display_img)
            k = cv2.waitKey(20) & 0xFF

            if k == 27: # ESC
                break
            elif k == ord('f'):
                self.mode = 1
                print("Brush: Foreground (White)")
            elif k == ord('b'):
                self.mode = 2
                print("Brush: Background (Red)")
            elif k == ord(' '): # Space
                self.run_segmentation()
            elif k == ord('r'):
                self.scribble_mask.fill(0)
                self.result_mask.fill(0)
                self.display_img = self.img.copy()
                print("Reset.")
            elif k == ord('s'):
                self.save_result()
            elif k == ord('m'):
                new_mode = 'hist' if self.model.mode == 'gmm' else 'gmm'
                self.model.mode = new_mode
                print(f"Switched to mode: {new_mode}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    InteractiveSegmenter()

