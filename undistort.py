"""
Fisheye lens undistortion for Donkeycar camera feeds.

Fill in FISHEYE_K and FISHEYE_D in myconfig.py after checkerboard calibration,
then set ENABLE_FISHEYE_UNDISTORT = True.
"""

import cv2
import numpy as np


class FisheyeCorrect:
    """
    Donkeycar Part: undistort a fisheye RGB frame using precomputed remap tables.

    Calibration placeholders below match IMAGE_W x IMAGE_H. Replace with values
    from your calibration script before enabling undistortion.
    """

    def __init__(self, cfg):
        self.dim = (cfg.IMAGE_W, cfg.IMAGE_H)

        # --- PLACEHOLDER: replace after fisheye calibration ---
        # 3x3 camera matrix K (fx, fy, cx, cy)
        default_k = np.array([
            [115.0, 0.0, cfg.IMAGE_W / 2.0],
            [0.0, 115.0, cfg.IMAGE_H / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # 4-element fisheye distortion coefficients (k1, k2, k3, k4)
        default_d = np.array([[-0.05, 0.01, 0.0, 0.0]], dtype=np.float64)

        k = getattr(cfg, 'FISHEYE_K', None)
        d = getattr(cfg, 'FISHEYE_D', None)
        self.K = np.asarray(k if k is not None else default_k, dtype=np.float64)
        self.D = np.asarray(d if d is not None else default_d, dtype=np.float64)

        if self.D.ndim == 1:
            self.D = self.D.reshape(1, 4)

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K, self.dim, cv2.CV_16SC2
        )

    def run(self, img):
        if img is None:
            return None
        return cv2.remap(
            img, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
