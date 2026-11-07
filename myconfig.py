"""
Active car overrides — merged on top of config.py by manage.py.
"""

# ---------------------------------------------------------------------------
# Camera preprocessing
# ---------------------------------------------------------------------------
CAMERA_ROTATE_180 = True
ENABLE_FISHEYE_UNDISTORT = False
FISHEYE_K = [
    [115.0, 0.0, 80.0],
    [0.0, 115.0, 60.0],
    [0.0, 0.0, 1.0],
]
FISHEYE_D = [-0.05, 0.01, 0.0, 0.0]

# ---------------------------------------------------------------------------
# Computer vision autopilot
# ---------------------------------------------------------------------------
USE_CV_AUTOPILOT = True

CV_CONTROLLER_MODULE = "cv_parts.lane_keeper"
CV_CONTROLLER_CLASS = "LaneKeeper"
# CV_CONTROLLER_MODULE = "cv_parts.line_follower"
# CV_CONTROLLER_CLASS = "LineFollower"
CV_CONTROLLER_INPUTS = ['cam/image_array']
CV_CONTROLLER_OUTPUTS = ['pilot/angle', 'pilot/throttle', 'cv/image_array']
CV_CONTROLLER_CONDITION = "run_pilot"

IMAGE_W = 160
IMAGE_H = 128   # PiCamera rounds 120 up to 128

import cv2

# LineFollower legacy settings (unused by LaneKeeper but kept for compatibility)
SCAN_Y = 95
SCAN_HEIGHT = 10
COLOR_THRESHOLD_LOW = (150, 100, 100)
COLOR_THRESHOLD_HIGH = (180, 255, 255)
TARGET_PIXEL = 80
TARGET_THRESHOLD = 10
CONFIDENCE_THRESHOLD = (1 / IMAGE_W) / 3

THROTTLE_MAX = 0.4
THROTTLE_MIN = 0.4
THROTTLE_INITIAL = THROTTLE_MIN
THROTTLE_STEP = 0.01

CV_PID_P = -12.0
CV_PID_I = 0.000
CV_PID_D = -0.25
CV_PID_P_DELTA = 0.1
CV_PID_D_DELTA = 0.01

OVERLAY_IMAGE = True

INC_PID_P_BTN = "R2"
DEC_PID_P_BTN = "L2"
TOGGLE_RECORDING_BTN = "option"

# LaneKeeper (switch CV_CONTROLLER_MODULE to cv_parts.lane_keeper to use)
LK_LANE_MODE = "yellow_white"
LK_CONFIDENCE_THRESHOLD = 0.0015
LK_TARGET_THRESHOLD = 10
LK_LANE_WIDTH = 60


# LaneKeeper
LK_SCAN_Y = 75
LK_SCAN_HEIGHT = 10

LK_TARGET_THRESHOLD = 10

# Identical HSV ranges to LineFollower
LK_HSV_LOW = (0, 80, 80)
LK_HSV_HIGH = (35, 255, 255)

# Histogram detection
LK_MIN_COLUMN_PIXELS = 3

# Allowed separation between two boundaries
LK_MIN_LANE_SEPARATION = 40
LK_MAX_LANE_SEPARATION = 155

# Permit markings near image edges
LK_EDGE_MARGIN = 2

# PID output normalization
LK_STEERING_SCALE = 500.0

# Safety
LK_RECOVERY_FRAMES = 3

# Diagnostics
LK_DEBUG = True
OVERLAY_IMAGE = True

# Keep initial throttle at zero for testing
THROTTLE_INITIAL = 0.0
AUTO_RECORD_ON_THROTTLE = False
RECORD_DURING_AI = False
