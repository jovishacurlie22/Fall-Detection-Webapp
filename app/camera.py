# ============================================================
#  Camera Module — initialise, read, release webcam cleanly
# ============================================================
import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class CameraStream:
    """Wraps OpenCV VideoCapture with clean init / teardown."""

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"❌  Cannot open camera at index {CAMERA_INDEX}. "
                "Check if your webcam is connected and not in use."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency
        print(f"✅  Camera opened — {FRAME_WIDTH}x{FRAME_HEIGHT}")

    def read(self):
        """Returns (success: bool, frame: np.ndarray)."""
        return self.cap.read()

    def release(self):
        self.cap.release()
        print("📷  Camera released.")