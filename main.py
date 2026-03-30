# ============================================================
#  Smart Elderly Fall Detection System
#  Phase 3 — Pose Estimation (MediaPipe)
# ============================================================
import cv2
import sys

from config            import WINDOW_TITLE, ACCENT_COLOR
from app.camera        import CameraStream
from app.detector      import HumanDetector
from app.pose_estimator import PoseEstimator
from app.utils         import (FPSCounter, draw_header, draw_footer,
                                draw_status_pill, draw_person_count,
                                draw_keypoint_debug)
from app.fall_detector import FallDetector


def main():
    print("=" * 58)
    print("  🛡️  Smart Fall Detection System  —  Phase 3")
    print("  Pose Estimation  |  MediaPipe Skeleton")
    print("=" * 58)
    print("  Press  Q  to quit\n")

    try:
        cam = CameraStream()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    detector      = HumanDetector()
    pose_estimator = PoseEstimator()
    fall_detector = FallDetector()
    fps_counter   = FPSCounter()

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        # ── Phase 2 — Person detection ────────────────────────
        frame, persons = detector.detect(frame)
        person_count   = len(persons)

        # ── Phase 3 — Pose estimation ─────────────────────────
        keypoints = None
        fall_detected = False

        if person_count > 0:
            frame, keypoints, _ = pose_estimator.process(frame)
            fall_detected = fall_detector.update(keypoints)


        fps = fps_counter.tick()

        # ── Pose status ───────────────────────────────────────
        pose_detected = keypoints is not None

        # ── HUD Overlays ──────────────────────────────────────
        draw_header(frame)
        draw_person_count(frame, person_count)
        draw_keypoint_debug(frame, keypoints)
        if fall_detected:
            cv2.putText(
                frame,
                "FALL SUSPECTED!",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )
            print("⚠️ FALL SUSPECTED")

        draw_footer(frame, fps,
                    status="POSE TRACKED" if pose_detected else
                           ("PERSON — NO POSE" if person_count else "SCANNING"))

        h, w = frame.shape[:2]
        badge_txt   = ("PHASE 3  |  POSE OK" if pose_detected
                       else "PHASE 3  |  NO POSE")
        badge_color = ACCENT_COLOR if pose_detected else (60, 60, 70)
        draw_status_pill(frame, badge_txt, badge_color, w - 250, 38)

        # ── Display ───────────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋  Quit signal received.")
            break

    pose_estimator.release()
    cam.release()
    cv2.destroyAllWindows()
    print("✅  Session ended cleanly.")


if __name__ == "__main__":
    main()
