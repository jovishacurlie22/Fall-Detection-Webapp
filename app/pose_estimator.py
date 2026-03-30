# ============================================================
#  Pose Estimator — MediaPipe Pose (Tasks API 0.10.30+)
# ============================================================
import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

KEYPOINT_INDICES = {
    "nose"          : 0,
    "left_shoulder" : 11,
    "right_shoulder": 12,
    "left_hip"      : 23,
    "right_hip"     : 24,
    "left_knee"     : 25,
    "right_knee"    : 26,
    "left_ankle"    : 27,
    "right_ankle"   : 28,
}

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)
]

MODEL_PATH = "pose_landmarker_lite.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥  Downloading MediaPipe pose model (~3 MB)...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        )
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("✅  Model downloaded.")


class PoseEstimator:

    def __init__(self):
        download_model()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        print("✅  MediaPipe PoseLandmarker loaded (lite model)")

    def process(self, frame: np.ndarray):
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            self._draw_no_pose(frame)
            return frame, None, None

        landmarks = result.pose_landmarks[0]

        self._draw_skeleton(frame, landmarks)
        self._draw_key_joints(frame, landmarks)

        keypoints = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in landmarks],
            dtype=np.float32
        ).flatten()

        return frame, keypoints, landmarks

    def _draw_skeleton(self, frame, landmarks):
        h, w = frame.shape[:2]

        for a, b in POSE_CONNECTIONS:
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            lm_a = landmarks[a]
            lm_b = landmarks[b]
            if lm_a.visibility < 0.4 or lm_b.visibility < 0.4:
                continue
            pt_a = (int(lm_a.x * w), int(lm_a.y * h))
            pt_b = (int(lm_b.x * w), int(lm_b.y * h))
            cv2.line(frame, pt_a, pt_b, (0, 180, 255), 2, cv2.LINE_AA)

        for lm in landmarks:
            if lm.visibility < 0.4:
                continue
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 140), -1, cv2.LINE_AA)

    def _draw_key_joints(self, frame, landmarks):
        h, w = frame.shape[:2]

        for name, idx in KEYPOINT_INDICES.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            if lm.visibility < 0.5:
                continue
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 7, (0, 255, 140), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_no_pose(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "NO POSE DETECTED — stand in view",
            (w // 2 - 220, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 60, 200),
            1,
            cv2.LINE_AA
        )

    def release(self):
        self.landmarker.close()