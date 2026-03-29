import argparse
import cv2
import numpy as np
import math

np.set_printoptions(suppress=True, precision=5)

'''
This file contains the implementation of homography estimation and image warping for image alignment.
It is an interactive homography script where you click correspondences manually.
'''

_clicked = None
_click_window_name = None

def _mouse_callback(event, x, y, flags, param):
    global _clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked = (int(x), int(y))

def pick_points(image, window_name='image', prompt="Click points; press 'a' to accept point; Esc or 'q' to finish"):
    global _clicked, _click_window_name
    _clicked = None
    _click_window_name = window_name
    pts = []

    img = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback)

    print(prompt)
    idx = 1
    while True:
        disp = img.copy()
        for i, p in enumerate(pts):
            cv2.drawMarker(disp, tuple(p), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(disp, str(i+1), (p[0]+8, p[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        if _clicked is not None:
            cv2.drawMarker(disp, _clicked, (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == ord('a'):
            if _clicked is None:
                print("No current click to accept. Click with left mouse button first.")
                continue
            pts.append(_clicked)
            print(f"Accepted point #{idx}: {_clicked}")
            idx += 1
            _clicked = None

    cv2.destroyWindow(window_name)
    return np.array(pts, dtype=float)


# Geometry helpers
def to_homogeneous(pts):
    pts = np.asarray(pts)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1))
    return np.hstack([pts, ones])

def from_homogeneous(pts_h):
    pts_h = np.asarray(pts_h)
    pts = pts_h[:, :2] / pts_h[:, 2:3]
    return pts

def normalize_points(pts):
    pts = np.asarray(pts)
    n = pts.shape[0]
    mean = pts.mean(axis=0)
    d = np.sqrt(((pts - mean)**2).sum(axis=1)).mean()
    scale = math.sqrt(2) / d
    T = np.array([
        [scale, 0, -scale*mean[0]],
        [0, scale, -scale*mean[1]],
        [0, 0, 1]
    ])
    pts_h = to_homogeneous(pts)
    pts_n = (T @ pts_h.T).T
    return pts_n, T

def dlt_homography(pts_src, pts_dst):
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)
    assert pts_src.shape == pts_dst.shape
    n = pts_src.shape[0]
    if n < 4:
        raise ValueError("At least 4 points required for homography")

    # Normalize
    src_n, T_src = normalize_points(pts_src)
    dst_n, T_dst = normalize_points(pts_dst)

    A = []
    for i in range(n):
        X = src_n[i]
        x, y, w = dst_n[i]
        Xx, Xy, Xw = X
        A.append([0,0,0, -w*Xx, -w*Xy, -w*Xw, y*Xx, y*Xy, y*Xw])
        A.append([w*Xx, w*Xy, w*Xw, 0,0,0, -x*Xx, -x*Xy, -x*Xw])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape((3,3))
    # Denormalize
    H = np.linalg.inv(T_dst) @ Hn @ T_src
    H = H / H[2,2]
    return H

def project_points(H, pts):
    pts_h = to_homogeneous(pts)
    pts_proj_h = (H @ pts_h.T).T
    pts_proj = from_homogeneous(pts_proj_h)
    return pts_proj

def symmetric_transfer_error(H, pts_src, pts_dst):
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)
    # forward
    proj = project_points(H, pts_src)
    err_f = np.sqrt(((proj - pts_dst)**2).sum(axis=1))
    # inverse
    H_inv = np.linalg.inv(H)
    proj_back = project_points(H_inv, pts_dst)
    err_b = np.sqrt(((proj_back - pts_src)**2).sum(axis=1))
    return (err_f + err_b) / 2.0


def main(args):
    # Load images
    imgA = cv2.imread(args.a)
    imgB = cv2.imread(args.b)
    imgC = None
    if args.c:
        imgC = cv2.imread(args.c)

    # pick correspondences for AB and BC
    print("\n--- Select at least 10 correspondences between A and B ---")
    ptsA = pick_points(imgA, window_name='Pick A->B points', prompt="Pick points in A; press 'a' to accept point; Esc/q when done.")
    print(f"Selected {len(ptsA)} points in A.")
    ptsB = pick_points(imgB, window_name='Pick B points', prompt="Pick matching points in B (same order).")
    if len(ptsA) != len(ptsB):
        nmin = min(len(ptsA), len(ptsB))
        ptsA = ptsA[:nmin]; ptsB = ptsB[:nmin]


    # linear DLT and symmetric error
    H_lin = dlt_homography(ptsA, ptsB)
    errs_lin = symmetric_transfer_error(H_lin, ptsA, ptsB)
    print("\nDLT homography (from all points):\n", H_lin)
    print("Mean symmetric transfer error (DLT):", np.mean(errs_lin))
    print("Per-point errors:", errs_lin)


    # warp B into A (inverse transform)
    hA, wA = imgA.shape[:2]
    H_BA = np.linalg.inv(H_lin)
    stitched = cv2.warpPerspective(imgB, H_BA, (wA + imgB.shape[1], max(hA, imgB.shape[0])))
    stitched[0:hA, 0:wA] = imgA
    cv2.namedWindow("Warped B into A", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped B into A", stitched)
    cv2.waitKey(1)

    if imgC is not None:
        print("\n--- Select correspondences between B and C (for chaining) ---")
        ptsB2 = pick_points(imgB, window_name='Pick B->C points', prompt="Pick points in B for B->C correspondences.")
        ptsC = pick_points(imgC, window_name='Pick C points for B->C', prompt="Pick matching points in C (same order).")
        nmin = min(len(ptsB2), len(ptsC))
        ptsB2 = ptsB2[:nmin]; ptsC = ptsC[:nmin]
        H_BC = dlt_homography(ptsB2, ptsC)
        print("\nH_BC (DLT):\n", H_BC)
        errs_BC = symmetric_transfer_error(H_BC, ptsB2, ptsC)
        print("Mean symmetric transfer error (BC DLT):", np.mean(errs_BC))

        # Compose H_AC_comp = H_BC * H_AB
        H_AC_comp = H_BC @ H_lin
        print("\nComposed H_AC (H_BC * H_AB):\n", H_AC_comp / H_AC_comp[2,2])

        ### Stitch Panorama (A -> B -> C)
        # Determine bounding box in C's coordinate frame
        def get_bbox(img, H):
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
            pts = cv2.perspectiveTransform(corners, H)
            return pts

        ptsA_in_C = get_bbox(imgA, H_AC_comp)
        ptsB_in_C = get_bbox(imgB, H_BC)
        ptsC_in_C = get_bbox(imgC, np.eye(3)) # Identity for C

        # Combine all corners to find min/max
        all_pts = np.concatenate((ptsA_in_C, ptsB_in_C, ptsC_in_C), axis=0)
        xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        W_p, H_p = xmax - xmin, ymax - ymin

        # Translation matrix to shift the top-left to (0,0)
        H_trans = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]], dtype=float)

        print(f"Computed panorama size: {W_p}x{H_p}")

        # Warp images into the panorama canvas
        canvas = np.zeros((H_p, W_p, 3), dtype=np.uint8)

        # Warp A
        wa = cv2.warpPerspective(imgA, H_trans @ H_AC_comp, (W_p, H_p))
        ma = cv2.warpPerspective(np.ones(imgA.shape[:2], np.uint8)*255, H_trans @ H_AC_comp, (W_p, H_p))
        canvas[ma > 0] = wa[ma > 0]

        # Warp B (overwrite A)
        wb = cv2.warpPerspective(imgB, H_trans @ H_BC, (W_p, H_p))
        mb = cv2.warpPerspective(np.ones(imgB.shape[:2], np.uint8)*255, H_trans @ H_BC, (W_p, H_p))
        canvas[mb > 0] = wb[mb > 0]

        # Warp C (overwrite B)
        wc = cv2.warpPerspective(imgC, H_trans, (W_p, H_p))
        mc = cv2.warpPerspective(np.ones(imgC.shape[:2], np.uint8)*255, H_trans, (W_p, H_p))
        canvas[mc > 0] = wc[mc > 0]

        cv2.namedWindow("Panorama ABC", cv2.WINDOW_NORMAL)
        cv2.imshow("Panorama ABC", canvas)
        if args.out_prefix:
            cv2.imwrite(args.out_prefix + "_panorama_ABC.jpg", canvas)

    # Save outputs
    if args.out_prefix:
        cv2.imwrite(args.out_prefix + "_warpedAtoB.png", stitched)
        print("Saved output images with prefix:", args.out_prefix)

    print("\nDone. Press any key in any image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="./data/A.png", help="Image A path")
    parser.add_argument("--b", type=str, default="./data/B.png", help="Image B path")
    parser.add_argument("--c", type=str, default="./data/C.png", help="Image C path (optional)")
    parser.add_argument("--out_prefix", type=str, default="./out", help="Prefix to save outputs")
    parser.add_argument("--ransac_thresh", type=float, default=4.0, help="RANSAC inlier threshold (pixels)")
    parser.add_argument("--ransac_iters", type=int, default=2000, help="RANSAC iterations")
    args = parser.parse_args()
    main(args)
