# ============================================================
# Phase 4 — Rule-Based Fall Detection (FIXED FOR FLATTENED KP)
# ============================================================

import time
import numpy as np


class FallDetector:

    def __init__(self):
        self.prev_height = None
        self.last_fall_time = 0
        self.cooldown = 3

    # ---------------------------------------------------------
    def _get_point(self, keypoints, idx):
        """Extract (x,y,visibility) from flattened array"""
        base = idx * 3
        return (
            keypoints[base],
            keypoints[base + 1],
            keypoints[base + 2],
        )

    # ---------------------------------------------------------
    def _body_height(self, keypoints):
        try:
            _, nose_y, v1 = self._get_point(keypoints, 0)
            _, la_y, v2 = self._get_point(keypoints, 27)
            _, ra_y, v3 = self._get_point(keypoints, 28)

            if min(v1, v2, v3) < 0.4:
                return None

            ankle_y = max(la_y, ra_y)
            return abs(ankle_y - nose_y)

        except Exception:
            return None

    # ---------------------------------------------------------
    def _is_horizontal(self, keypoints):

        try:
            _, ls_y, v1 = self._get_point(keypoints, 11)
            _, rs_y, v2 = self._get_point(keypoints, 12)
            _, lh_y, v3 = self._get_point(keypoints, 23)
            _, rh_y, v4 = self._get_point(keypoints, 24)

            if min(v1, v2, v3, v4) < 0.4:
                return False

            shoulder_avg = (ls_y + rs_y) / 2
            hip_avg = (lh_y + rh_y) / 2

            # relaxed threshold (normalized coords)
            return abs(shoulder_avg - hip_avg) < 0.12

        except Exception:
            return False

    # ---------------------------------------------------------
    def update(self, keypoints):

        if keypoints is None:
            return False

        height = self._body_height(keypoints)
        if height is None:
            return False

        fall_detected = False

        if self.prev_height is not None:

            drop_ratio = height / (self.prev_height + 1e-6)

            sudden_drop = drop_ratio < 0.8
            horizontal = self._is_horizontal(keypoints)

            print(
                f"[DEBUG] height={height:.3f} "
                f"ratio={drop_ratio:.2f} "
                f"horizontal={horizontal}"
            )

            if sudden_drop and horizontal:
                now = time.time()
                if now - self.last_fall_time > self.cooldown:
                    fall_detected = True
                    self.last_fall_time = now

        self.prev_height = height
        return fall_detected
