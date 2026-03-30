# ============================================================
#  Smart Elderly Fall Detection System
#  Phase 2 — Human Detection (YOLOv8n)
# ============================================================
import cv2
import sys

from config       import WINDOW_TITLE, ACCENT_COLOR
from app.camera   import CameraStream
from app.detector import HumanDetector
from app.utils    import (FPSCounter, draw_header, draw_footer,
                          draw_status_pill, draw_person_count)


def main():
    print("=" * 58)
    print("  🛡️  Smart Fall Detection System  —  Phase 2")
    print("  Human Detection  |  YOLOv8 Nano")
    print("=" * 58)
    print("  Press  Q  to quit\n")

    try:
        cam = CameraStream()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    detector    = HumanDetector()
    fps_counter = FPSCounter()

    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        # ── Detection ────────────────────────────────────────
        frame, persons = detector.detect(frame)
        person_count   = len(persons)
        fps            = fps_counter.tick()

        # ── HUD Overlays ──────────────────────────────────────
        draw_header(frame)
        draw_person_count(frame, person_count)
        draw_footer(frame, fps,
                    status="PERSON DETECTED" if person_count else "NO PERSON")

        # Phase badge (top-right)
        h, w = frame.shape[:2]
        badge_color = ACCENT_COLOR if person_count else (60, 60, 70)
        draw_status_pill(frame,
                         f"PHASE 2  |  YOLO  |  {person_count} PERSON(S)",
                         badge_color, w - 310, 38)

        # ── Display ───────────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋  Quit signal received.")
            break

    cam.release()
    cv2.destroyAllWindows()
    print("✅  Session ended cleanly.")


if __name__ == "__main__":
    main()
