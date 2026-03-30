# ============================================================
#  Human Detector — YOLOv8 Nano (Person class only)
#  Why YOLOv8n: fastest YOLO variant, ~3ms CPU inference,
#               pretrained on COCO (class 0 = person).
# ============================================================
from ultralytics import YOLO
import cv2
import numpy as np


# ── Visual style constants ───────────────────────────────────
BOX_COLOR       = (0, 220, 110)      # vivid green  (BGR)
BOX_THICKNESS   = 2
LABEL_BG_COLOR  = (0, 220, 110)
LABEL_TXT_COLOR = (10, 10, 10)       # near-black for contrast
CONF_THRESHOLD  = 0.45               # ignore detections below 45 %


class HumanDetector:
    """
    Wraps YOLOv8n and filters detections to 'person' class only.
    Draws sleek bounding boxes + confidence labels on the frame.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        print("🔍  Loading YOLOv8n … (downloads ~6 MB on first run)")
        self.model = YOLO(model_path)
        self.person_class_id = 0        # COCO class 0 = person
        print("✅  YOLOv8n loaded.")

    # ────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run inference on one frame.

        Returns
        -------
        annotated_frame : frame with bounding boxes drawn
        persons         : list of dicts  {box, conf, centre}
        """
        results = self.model(
            frame,
            classes=[self.person_class_id],   # filter to person only
            conf=CONF_THRESHOLD,
            verbose=False                      # suppress per-frame logs
        )[0]

        persons = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cx    = (x1 + x2) // 2
            cy    = (y1 + y2) // 2

            persons.append({
                "box"    : (x1, y1, x2, y2),
                "conf"   : conf,
                "centre" : (cx, cy),
            })

            self._draw_box(frame, x1, y1, x2, y2, conf)

        return frame, persons

    # ────────────────────────────────────────────────────────
    def _draw_box(self, frame, x1, y1, x2, y2, conf):
        """Draw a clean bounding box with corner accents + label."""
        # Main rectangle (thin, bright green)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      BOX_COLOR, BOX_THICKNESS, cv2.LINE_AA)

        # Corner accent marks (thicker, for a tactical-HUD look)
        arm = 18
        t   = 3
        corners = [
            ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),   # top-left
            ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),   # top-right
            ((x1, y2 - arm), (x1, y2), (x1 + arm, y2)),   # bottom-left
            ((x2 - arm, y2), (x2, y2), (x2, y2 - arm)),   # bottom-right
        ]
        for pts in corners:
            cv2.line(frame, pts[0], pts[1], BOX_COLOR, t, cv2.LINE_AA)
            cv2.line(frame, pts[1], pts[2], BOX_COLOR, t, cv2.LINE_AA)

        # Confidence label pill
        label = f" PERSON  {conf:.0%} "
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame,
                      (x1, y1 - lh - 10),
                      (x1 + lw, y1),
                      LABEL_BG_COLOR, -1, cv2.LINE_AA)
        cv2.putText(frame, label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    LABEL_TXT_COLOR, 1, cv2.LINE_AA)

        # Centre crosshair dot
        cv2.circle(frame,
                   ((x1 + x2) // 2, (y1 + y2) // 2),
                   4, BOX_COLOR, -1, cv2.LINE_AA)