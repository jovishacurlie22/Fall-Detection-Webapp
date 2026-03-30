# ============================================================
#  Smart Elderly Fall Detection — Global Configuration
# ============================================================

# Camera
CAMERA_INDEX = 0          # 0 = default webcam
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS_TARGET   = 30

# Display
WINDOW_TITLE = "🛡️ Smart Fall Detection System — Live Monitor"
FONT_SCALE   = 0.65
FONT_COLOR   = (255, 255, 255)   # white
ACCENT_COLOR = (0, 200, 100)     # green
WARN_COLOR   = (0, 100, 255)     # orange-red (BGR)
DANGER_COLOR = (0, 0, 220)       # red (BGR)

# Paths
MODEL_PATH   = "models/fall_lstm.pth"
LOG_PATH     = "logs/alerts.log"