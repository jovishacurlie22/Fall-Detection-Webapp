# ============================================================
#  Utility helpers — overlays, FPS counter, status bar
# ============================================================
import cv2
import time


class FPSCounter:
    """Smooth FPS tracker using rolling average."""

    def __init__(self, smoothing: int = 30):
        self._times = []
        self._smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._smoothing:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def draw_header(frame, title: str = "SMART FALL DETECTION SYSTEM"):
    """Dark semi-transparent header bar with project title."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent line
    cv2.line(frame, (0, 54), (w, 54), (0, 200, 100), 2)

    # Shield icon placeholder dot
    cv2.circle(frame, (32, 27), 12, (0, 200, 100), -1)
    cv2.circle(frame, (32, 27), 12, (255, 255, 255), 1)

    cv2.putText(frame, title,
                (56, 35), cv2.FONT_HERSHEY_DUPLEX,
                0.72, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status_pill(frame, text: str, color: tuple, x: int, y: int):
    """Rounded-corner status pill (simulated with rectangle)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    pad = 10
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + pad),
                  color, -1, cv2.LINE_AA)
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + pad),
                  (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text,
                (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 255, 255), 1, cv2.LINE_AA)


def draw_footer(frame, fps: float, status: str = "MONITORING"):
    """Bottom HUD bar — FPS, resolution, status."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, h - 40), (w, h - 40), (0, 200, 100), 1)

    left  = f"  FPS: {fps:5.1f}"
    right = f"RES: {w}x{h}  |  STATUS: {status}  "
    cv2.putText(frame, left,
                (8, h - 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, right,
                (w - 340, h - 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (180, 180, 180), 1, cv2.LINE_AA)


def draw_person_count(frame, count: int):
    """Top-left detection count badge."""
    h, w = frame.shape[:2]
    color  = (0, 200, 100) if count > 0 else (80, 80, 90)
    label  = f"  PERSONS DETECTED: {count}  "
    cv2.putText(frame, label,
                (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                color, 1, cv2.LINE_AA)
    # horizontal divider
    cv2.line(frame, (0, 90), (w, 90), (40, 40, 45), 1)