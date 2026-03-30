# ============================================================
#  Smart Elderly Fall Detection System
#  Phase 1 — Live Camera Monitor
# ============================================================
import cv2
import sys

from config        import WINDOW_TITLE, ACCENT_COLOR
from app.camera    import CameraStream
from app.utils     import FPSCounter, draw_header, draw_footer, draw_status_pill


def main():
    print("=" * 58)
    print("  🛡️  Smart Fall Detection System  —  Phase 1")
    print("  Live Camera Monitor")
    print("=" * 58)
    print("  Press  Q  to quit\n")

    try:
        cam = CameraStream()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    fps_counter = FPSCounter()

    while True:
        ok, frame = cam.read()
        if not ok:
            print("⚠️  Frame grab failed — retrying…")
            continue

        fps = fps_counter.tick()

        # ── Overlays ─────────────────────────────────────────
        draw_header(frame)
        draw_footer(frame, fps, status="MONITORING")

        # Phase badge (top-right)
        h, w = frame.shape[:2]
        draw_status_pill(frame, "PHASE 1  |  CAMERA OK",
                         ACCENT_COLOR, w - 230, 38)

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
