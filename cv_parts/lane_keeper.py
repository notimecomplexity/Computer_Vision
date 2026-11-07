import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class LaneKeeper:
    """
    Lane keeping using the same HSV sensing as LineFollower.

    Scan several horizontal bands, in priority order.
    Use the first band where both boundaries are visible.
    If no band has both, fall back to the first band with
    one boundary and offset by half that band's lane width.
    Stop only when no band has a usable boundary.
    """

    def __init__(self, pid, cfg):
        self.pid_st = pid

        self.image_w = cfg.IMAGE_W
        self.scan_y = cfg.LK_SCAN_Y
        self.scan_height = cfg.LK_SCAN_HEIGHT

        self.overlay_image = cfg.OVERLAY_IMAGE
        self.debug = getattr(cfg, "LK_DEBUG", False)

        self.target_pixel = self.image_w // 2
        self.target_threshold = cfg.LK_TARGET_THRESHOLD

        self.steering = 0.0
        self.throttle = 0.0

        self.throttle_min = cfg.THROTTLE_MIN
        self.throttle_max = cfg.THROTTLE_MAX
        self.delta_th = cfg.THROTTLE_STEP

        # Defaults match the LineFollower HSV detection.
        self.hsv_low = np.array(
            getattr(cfg, "LK_HSV_LOW", (0, 80, 80)),
            dtype=np.uint8
        )
        self.hsv_high = np.array(
            getattr(cfg, "LK_HSV_HIGH", (35, 255, 255)),
            dtype=np.uint8
        )
        self.hsv_red_low = np.array([170, 80, 80],
                                    dtype=np.uint8)
        self.hsv_red_high = np.array([179, 255, 255],
                                     dtype=np.uint8)

        # At least this many matching pixels per column.
        self.min_column_pixels = getattr(
            cfg, "LK_MIN_COLUMN_PIXELS", 3
        )

        self.min_separation = getattr(
            cfg, "LK_MIN_LANE_SEPARATION", 40
        )
        self.max_separation = getattr(
            cfg, "LK_MAX_LANE_SEPARATION", 155
        )

        self.edge_margin = getattr(
            cfg, "LK_EDGE_MARGIN", 2
        )

        # Fixed steering normalization, independent of Kp.
        self.steering_scale = getattr(
            cfg, "LK_STEERING_SCALE", 960.0
        )

        self.recovery_frames = getattr(
            cfg, "LK_RECOVERY_FRAMES", 3
        )
        self.good_frames = 0
        self.last_debug = 0.0

        # Scan bands (top y of each), in priority order.
        # Default: the main band, then alternating above
        # and below it, one band height apart.
        step = getattr(cfg, "LK_SCAN_STEP", self.scan_height)
        n = getattr(cfg, "LK_SCAN_BANDS", 5)
        default_ys = [self.scan_y]
        k = 1
        while len(default_ys) < n:
            default_ys.append(self.scan_y - k * step)
            if len(default_ys) < n:
                default_ys.append(self.scan_y + k * step)
            k += 1
        self.scan_ys = [int(y) for y in
                        getattr(cfg, "LK_SCAN_YS", default_ys)]

        # Lane width differs by row because of perspective,
        # so each band keeps its own estimate and history.
        init_width = float(getattr(
            cfg, "LK_LANE_WIDTH",
            (self.min_separation + self.max_separation) / 2.0
        ))
        self.width_alpha = getattr(cfg, "LK_WIDTH_ALPHA", 0.2)
        self.band_width = {y: init_width for y in self.scan_ys}
        self.band_center = {y: None for y in self.scan_ys}

        self.pid_st.setpoint = self.target_pixel

    def _get_mask(self, scan_line):
        """Same color thresholding as LineFollower."""
        hsv = cv2.cvtColor(
            scan_line, cv2.COLOR_RGB2HSV
        )

        mask_lo = cv2.inRange(
            hsv, self.hsv_low, self.hsv_high
        )
        mask_hi = cv2.inRange(
            hsv, self.hsv_red_low, self.hsv_red_high
        )

        return cv2.bitwise_or(mask_lo, mask_hi)

    def _histogram_peak(self, mask, start, end):
        """
        Find the strongest colored column in a region.

        Return (x position, support) or (None, 0).
        Support is the average number of matching
        pixels across a small horizontal window.
        """
        region = mask[:, start:end]

        if region.size == 0:
            return None, 0.0

        # Count white pixels, not their 0/255 intensity.
        hist = np.count_nonzero(region, axis=0)

        # Smooth over five columns to reduce noise.
        kernel = np.ones(5, dtype=float) / 5.0
        smooth = np.convolve(hist, kernel, mode="same")

        peak = int(np.argmax(smooth))
        support = float(smooth[peak])

        if support < self.min_column_pixels:
            return None, support

        return start + peak, support

    def _valid_image(self, cam_img):
        return (isinstance(cam_img, np.ndarray)
                and cam_img.ndim == 3
                and cam_img.shape[2] == 3
                and cam_img.dtype == np.uint8
                and cam_img.shape[1] == self.image_w
                and self.scan_height >= 1)

    def detect_lines(self, cam_img, y):
        """
        Detect left/right boundaries in the band at y.

        Return (left_x, right_x, mask, left_score, right_score),
        or None if the band falls outside the image.
        """
        if y < 0 or y + self.scan_height > cam_img.shape[0]:
            return None

        scan_line = cam_img[y:y + self.scan_height]
        mask = self._get_mask(scan_line)

        mid = self.image_w // 2

        left_x, left_score = self._histogram_peak(
            mask, 0, mid
        )
        right_x, right_score = self._histogram_peak(
            mask, mid, self.image_w
        )

        # Reject detections near the image edges.
        if (left_x is not None and
                left_x < self.edge_margin):
            left_x = None

        if (right_x is not None and
                right_x >= self.image_w - self.edge_margin):
            right_x = None

        return left_x, right_x, mask, left_score, right_score

    def _valid_pair(self, left_x, right_x):
        return (left_x is not None and right_x is not None
                and self.min_separation <= right_x - left_x
                <= self.max_separation)

    def _single_edge(self, y, left_x, right_x):
        """
        Estimate the center in band y from one boundary.

        Return (center, est_left, est_right, side) or None.
        """
        width = self.band_width[y]
        last_c = self.band_center[y]

        # Implausible pair: keep the edge that best matches
        # history, or skip this band without history.
        if left_x is not None and right_x is not None:
            if last_c is None:
                return None
            if (abs(left_x - (last_c - width / 2)) <=
                    abs(right_x - (last_c + width / 2))):
                right_x = None
            else:
                left_x = None

        edge = left_x if left_x is not None else right_x
        if edge is None:
            return None

        # The histogram splits at the image center, so a line
        # that drifted past the middle lands in the wrong half.
        # Prefer history over the half it was found in.
        is_left = left_x is not None
        if last_c is not None:
            is_left = edge <= last_c

        if is_left:
            est_left = float(edge)
            est_right = est_left + width
            side = "L"
        else:
            est_right = float(edge)
            est_left = est_right - width
            side = "R"

        center = (est_left + est_right) / 2.0
        self.band_center[y] = center
        return center, est_left, est_right, side

    def _stop(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.good_frames = 0

        # Forget edge history so a stale position does not
        # mislabel the first edge seen after recovery.
        # Lane widths are kept since they are good priors.
        for y in self.band_center:
            self.band_center[y] = None

        reset = getattr(self.pid_st, "reset", None)
        if callable(reset):
            reset()
        self.pid_st.setpoint = self.target_pixel

    def run(self, cam_img):
        """
        Main control loop.

        Return (steering, throttle, output_image).
        """
        if cam_img is None or not self._valid_image(cam_img):
            self._stop()
            return 0.0, 0.0, cam_img

        # Detect in every band that fits in the image.
        bands = []
        for y in self.scan_ys:
            det = self.detect_lines(cam_img, y)
            if det is not None:
                bands.append((y,) + det)

        lane_center = None
        est_left = est_right = None
        chosen = None
        status = "LOST"

        # Pass 1: first band with both edges. Every band with
        # a good pair also refines its own width and history.
        for y, l, r, _, _, _ in bands:
            if not self._valid_pair(l, r):
                continue
            a = self.width_alpha
            self.band_width[y] = ((1 - a) * self.band_width[y]
                                  + a * (r - l))
            self.band_center[y] = (l + r) / 2.0
            if chosen is None:
                chosen = y
                lane_center = (l + r) / 2.0
                est_left, est_right = float(l), float(r)
                status = "TRACKING"

        # Pass 2: first band with at least one usable edge.
        if chosen is None:
            for y, l, r, _, _, _ in bands:
                res = self._single_edge(y, l, r)
                if res is not None:
                    chosen = y
                    lane_center, est_left, est_right, side = res
                    status = "ONE EDGE ({})".format(side)
                    break

        if lane_center is None:
            self._stop()

        else:
            self.good_frames += 1

            if self.good_frames < self.recovery_frames:
                self.steering = 0.0
                self.throttle = 0.0
                status = "RECOVERING"

            else:
                raw = float(self.pid_st(lane_center))

                if not np.isfinite(raw):
                    self._stop()
                    status = "INVALID PID"
                else:
                    self.steering = float(
                        np.tanh(raw / self.steering_scale)
                    )

                    error = abs(
                        lane_center - self.target_pixel
                    )

                    if error > self.target_threshold:
                        self.throttle = max(
                            self.throttle_min,
                            self.throttle - self.delta_th
                        )
                    else:
                        self.throttle = min(
                            self.throttle_max,
                            self.throttle + self.delta_th
                        )

                    # Ramp up from zero after recovery.
                    if self.throttle == 0.0:
                        self.throttle = min(
                            self.delta_th,
                            self.throttle_max
                        )

        if self.debug:
            now = time.monotonic()
            if now - self.last_debug >= 1.0:
                found = " ".join(
                    "y{}:{}{}".format(
                        y,
                        "L" if l is not None else "-",
                        "R" if r is not None else "-"
                    )
                    for y, l, r, _, _, _ in bands
                )
                width = (self.band_width[chosen]
                         if chosen is not None else float("nan"))
                print(
                    "[LaneKeeper] {} band={} [{}] W={:.1f} "
                    "center={} ST={:.3f} TH={:.3f}"
                    .format(
                        status, chosen, found, width,
                        lane_center, self.steering, self.throttle
                    ),
                    flush=True
                )
                self.last_debug = now

        if self.overlay_image and bands:
            cam_img = self.overlay_display(
                cam_img, bands, chosen,
                est_left, est_right, lane_center, status
            )

        return float(self.steering), float(self.throttle), cam_img

    def overlay_display(self, cam_img, bands, chosen,
                        est_left, est_right,
                        lane_center, status):
        img = cam_img.copy()
        h = self.scan_height

        # Show the binary mask in every scanned band, and
        # the detected boundaries for each band in green.
        for y, l, r, mask, _, _ in bands:
            img[y:y + h] = cv2.cvtColor(
                mask, cv2.COLOR_GRAY2RGB
            )
            for x in (l, r):
                if x is not None:
                    cv2.circle(
                        img, (int(x), y + h // 2),
                        2, (0, 255, 0), -1
                    )

        if chosen is not None:
            y = chosen
            scan_mid = y + h // 2

            # Outline the band actually used for steering.
            cv2.rectangle(
                img, (0, y), (self.image_w - 1, y + h - 1),
                (255, 0, 255), 1
            )

            # Inferred boundaries in the chosen band: yellow.
            for x in (est_left, est_right):
                if x is not None and 0 <= x < self.image_w:
                    cv2.circle(
                        img, (int(x), scan_mid),
                        3, (255, 255, 0), 1
                    )

            # Lane center: red.
            if lane_center is not None:
                cv2.circle(
                    img, (int(lane_center), scan_mid),
                    3, (255, 0, 0), -1
                )

            # Desired target: blue.
            cv2.line(
                img, (self.target_pixel, y),
                (self.target_pixel, y + h),
                (0, 0, 255), 1
            )

        cv2.putText(
            img, status, (3, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (255, 255, 255), 1
        )

        return img