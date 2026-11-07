"""
Camera image preprocessing for Donkeycar.

Applies orientation correction and optional fisheye undistortion before
images reach CV autopilot, recording, or the web portal.
"""

import cv2

from undistort import FisheyeCorrect


class ImagePreprocessor:
    """
    Donkeycar Part: cam/raw_image_array -> cam/image_array

    Processing order:
      1. 180-degree rotation (if CAMERA_ROTATE_180)
      2. Fisheye undistortion (if ENABLE_FISHEYE_UNDISTORT)
    """

    def __init__(self, cfg):
        self.rotate_180 = getattr(cfg, 'CAMERA_ROTATE_180', False)
        self.fisheye = None
        if getattr(cfg, 'ENABLE_FISHEYE_UNDISTORT', False):
            self.fisheye = FisheyeCorrect(cfg)

    def run(self, img):
        if img is None:
            return None
        if self.rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if self.fisheye is not None:
            img = self.fisheye.run(img)
        return img
