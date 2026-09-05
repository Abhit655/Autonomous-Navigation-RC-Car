"""
lane_pid_controller.py
----------------------
Python port of PidLaneFollower.cs from the Unity digital twin. Same algorithm, same tuning
parameter names, so gains found in simulation transfer directly to the car:

    SegFormer class mask
      -> per-row lane-line extraction  (find the two lines bracketing the car)
      -> lateral offset + heading measurement
      -> Kalman filter                 (smooths noise, coasts through missed detections)
      -> PID                           (+ heading feed-forward)
      -> steering command in -1..+1

Nothing here touches the camera, TensorRT or I2C - it is pure logic operating on a
numpy class-ID mask, which makes it identical in Unity and on the Jetson and easy to test
offline against recorded frames.

Two hard-won details carried over from the Unity work and Sean's thesis:

1. Lane extraction works on contiguous RUNS of line pixels, not individual pixels. A naive
   "nearest line pixel left and right of centre" scan returns the two EDGES OF THE SAME LINE
   whenever a line sits near the image centre, reporting an ~8px wide "lane" and sending the
   controller hard over. min_lane_width_px rejects those.

2. The Kalman filter must not integrate forever when no lines are visible - unbounded
   prediction drives the state to absurd values, saturates the steering, and puts the car
   off the track where it can never recover. Hence Kalman.coast() and the blind-reset.
   (Sean's fix for the same failure was a moving-average window; this is its principled cousin.)
"""

import numpy as np


class Kalman2:
    """Minimal 2-state (position, velocity) Kalman filter with position-only measurements."""

    def __init__(self, process_noise, measurement_noise):
        self.q = process_noise
        self.r = measurement_noise
        self.reset()

    def reset(self):
        self.p = 0.0   # position (the estimated quantity)
        self.v = 0.0   # rate of change
        self.p00, self.p01, self.p10, self.p11 = 1.0, 0.0, 0.0, 1.0

    def predict(self, dt):
        self.p += self.v * dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        n00 = self.p00 + dt * (self.p10 + self.p01) + dt2 * self.p11 + self.q * dt3 / 3.0
        n01 = self.p01 + dt * self.p11 + self.q * dt2 / 2.0
        n10 = self.p10 + dt * self.p11 + self.q * dt2 / 2.0
        n11 = self.p11 + self.q * dt
        self.p00, self.p01, self.p10, self.p11 = n00, n01, n10, n11

    def update(self, z):
        s = self.p00 + self.r
        k0 = self.p00 / s
        k1 = self.p10 / s
        innovation = z - self.p
        self.p += k0 * innovation
        self.v += k1 * innovation
        n00 = (1.0 - k0) * self.p00
        n01 = (1.0 - k0) * self.p01
        n10 = self.p10 - k1 * self.p00
        n11 = self.p11 - k1 * self.p01
        self.p00, self.p01, self.p10, self.p11 = n00, n01, n10, n11

    def coast(self, limit, velocity_decay):
        """Bound the state while blind so prediction cannot run away."""
        self.v *= velocity_decay
        self.p = max(-limit, min(limit, self.p))
        self.v = max(-limit, min(limit, self.v))


class LanePidController:
    def __init__(self,
                 # --- mask scanning ---
                 scan_band_top=0.45,       # fraction of image height; rows above this are ignored
                 scan_band_bottom=0.95,
                 scan_row_step=3,
                 min_lane_width_px=18,     # reject rows whose "lane" is implausibly narrow
                 line_class_ids=(2, 3, 4, 5, 6, 7),   # divider/dotted/double/random/road-sign/solid
                 # --- Kalman ---
                 process_noise=4.0,
                 measurement_noise=0.03,
                 # --- PID ---
                 kp=0.6, ki=0.05, kd=0.7,
                 heading_gain=1.8,
                 integral_limit=0.4,
                 offset_trim=0.0,          # corrects a systematic left/right bias
                 steering_deadband=0.04,   # ignore tiny errors so noise doesn't cause twitching
                 # --- throttle ---
                 base_throttle=0.45,
                 corner_slowdown=0.7,
                 lost_line_throttle=0.0,   # 0 = stop when blind. Safer default than the sim's crawl.
                 blind_updates_before_reset=25):
        self.scan_band_top = scan_band_top
        self.scan_band_bottom = scan_band_bottom
        self.scan_row_step = scan_row_step
        self.min_lane_width_px = min_lane_width_px
        self.line_classes = np.array(sorted(line_class_ids), dtype=np.uint8)

        self.kp, self.ki, self.kd = kp, ki, kd
        self.heading_gain = heading_gain
        self.integral_limit = integral_limit
        self.offset_trim = offset_trim
        self.steering_deadband = steering_deadband

        self.base_throttle = base_throttle
        self.corner_slowdown = corner_slowdown
        self.lost_line_throttle = lost_line_throttle
        self.blind_updates_before_reset = blind_updates_before_reset

        self.kf_offset = Kalman2(process_noise, measurement_noise)
        self.kf_heading = Kalman2(process_noise, measurement_noise)

        self.integral = 0.0
        self.lane_width_px = -1.0
        self.blind_updates = 0

        # Telemetry, read by the caller for logging.
        self.lines_visible = False
        self.rows_found = 0
        self.offset = 0.0
        self.heading = 0.0
        self.steer = 0.0
        self.throttle = 0.0

    # --------------------------------------------------------------------------------
    def _row_lane_centre(self, row, centre_x, res):
        """
        For one image row, return (lane_centre_x, lane_width or None) or (None, None).

        Works on contiguous runs: a lane line is several pixels thick, so we find the INNER
        edge of the nearest run on each side of centre. If a run straddles the image centre
        (the car is sitting on top of a line) both searches begin outside that run instead,
        so we lock onto the real lane boundaries rather than measuring the line underneath us.
        """
        is_line = np.isin(row, self.line_classes)

        search_left_from = centre_x
        search_right_from = centre_x + 1

        if is_line[centre_x]:
            l = centre_x
            while l >= 0 and is_line[l]:
                l -= 1
            r = centre_x
            while r < res and is_line[r]:
                r += 1
            search_left_from = l
            search_right_from = r

        left = -1
        for x in range(search_left_from, -1, -1):
            if is_line[x]:
                left = x
                break

        right = -1
        for x in range(search_right_from, res):
            if is_line[x]:
                right = x
                break

        if left >= 0 and right >= 0 and (right - left) >= self.min_lane_width_px:
            return 0.5 * (left + right), float(right - left)

        # Only one line usable: infer the centre from the remembered lane width.
        if left >= 0 and self.lane_width_px > 0:
            return left + self.lane_width_px * 0.5, None
        if right >= 0 and self.lane_width_px > 0:
            return right - self.lane_width_px * 0.5, None

        return None, None

    def extract_measurement(self, mask):
        """
        mask: (H, W) uint8 array of class IDs, row 0 = TOP of the camera image.
        Returns (ok, offset_norm, heading_slope).
            offset_norm   -1 .. +1, positive = lane centre is RIGHT of image centre
                          (i.e. the car sits left of where it should be)
            heading_slope -1 .. +1, positive = lane bends right ahead
        """
        h, w = mask.shape
        centre_x = w // 2

        y_start = int(np.clip(self.scan_band_top * h, 0, h - 1))
        y_end = int(np.clip(self.scan_band_bottom * h, 0, h - 1))

        ys, centres = [], []
        width_sum, width_count = 0.0, 0

        for y in range(y_start, y_end + 1, max(1, self.scan_row_step)):
            c, width = self._row_lane_centre(mask[y], centre_x, w)
            if c is None:
                continue
            ys.append(float(y))
            centres.append(c)
            if width is not None:
                width_sum += width
                width_count += 1

        if width_count > 0:
            avg = width_sum / width_count
            self.lane_width_px = avg if self.lane_width_px < 0 else (
                self.lane_width_px + 0.1 * (avg - self.lane_width_px))

        self.rows_found = len(centres)
        if len(centres) < 3:
            return False, 0.0, 0.0

        ys = np.array(ys)
        centres = np.array(centres)

        # Offset: weighted average lane centre, biased toward the rows nearest the car.
        span = max(1.0, y_end - y_start)
        weights = 0.25 + 0.75 * ((ys - y_start) / span)
        lane_centre = float(np.sum(centres * weights) / np.sum(weights))
        offset_norm = float(np.clip((lane_centre - centre_x) / (w * 0.5), -1.0, 1.0))

        # Heading: least-squares slope of lane centre vs row. y grows DOWNWARD, so a lane
        # bending right produces a negative dx/dy - hence the sign flip. Scaled by the band
        # height and normalised so it lives in the same -1..1 range as the offset, otherwise
        # heading_gain would be multiplying a wildly different magnitude.
        var_y = float(np.sum((ys - ys.mean()) ** 2))
        heading_slope = 0.0
        if var_y > 1e-4:
            cov = float(np.sum((ys - ys.mean()) * (centres - centres.mean())))
            band_height = max(1.0, y_end - y_start)
            heading_slope = float(np.clip(-(cov / var_y) * band_height / (w * 0.5), -1.0, 1.0))

        return True, offset_norm, heading_slope

    # --------------------------------------------------------------------------------
    def step(self, mask, dt):
        """
        Run one control cycle. Returns (steer, throttle) with
            steer    -1 = full left, +1 = full right
            throttle  0 .. 1
        """
        self.kf_offset.predict(dt)
        self.kf_heading.predict(dt)

        ok, offset_meas, heading_meas = (False, 0.0, 0.0)
        if mask is not None:
            ok, offset_meas, heading_meas = self.extract_measurement(mask)

        if ok:
            self.kf_offset.update(offset_meas)
            self.kf_heading.update(heading_meas)
            self.lines_visible = True
            self.blind_updates = 0
        else:
            self.lines_visible = False
            self.blind_updates += 1
            self.kf_offset.coast(1.5, 0.9)
            self.kf_heading.coast(1.5, 0.9)
            if self.blind_updates > self.blind_updates_before_reset:
                self.kf_offset.reset()
                self.kf_heading.reset()
                self.integral = 0.0

        offset = self.kf_offset.p - self.offset_trim
        if abs(offset) < self.steering_deadband:
            offset = 0.0
        offset_rate = self.kf_offset.v
        heading = self.kf_heading.p

        if self.lines_visible:
            self.integral = float(np.clip(self.integral + offset * dt,
                                          -self.integral_limit, self.integral_limit))
        else:
            # Bleed the integral off while blind rather than accumulating a stale error.
            self.integral -= np.sign(self.integral) * min(abs(self.integral), dt)

        steer = (self.kp * offset
                 + self.ki * self.integral
                 + self.kd * offset_rate
                 + self.heading_gain * heading)
        steer = float(np.clip(steer, -1.0, 1.0))

        if self.lines_visible:
            throttle = self.base_throttle * (1.0 - self.corner_slowdown * abs(steer))
        else:
            throttle = self.lost_line_throttle
            steer = 0.0 if self.blind_updates > self.blind_updates_before_reset else steer

        self.offset = offset
        self.heading = heading
        self.steer = steer
        self.throttle = max(0.0, throttle)
        return self.steer, self.throttle

    def telemetry(self):
        return (f"offset={self.offset:+.3f} heading={self.heading:+.3f} "
                f"steer={self.steer:+.3f} thr={self.throttle:.2f} "
                f"rows={self.rows_found} lanepx={self.lane_width_px:.1f} "
                f"{'SEE' if self.lines_visible else 'BLIND'}")
