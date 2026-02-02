


import cv2
import streamlit as st
import time
import os

from core.detector import YOLODetector
from core.tracker import CentroidTracker
from core.face_recognizer import recognize
from core.cache import IdentityCache
from core.line_logic import crossed_line

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Core_Model_1.pt")
# ------------------------------------------

st.set_page_config(layout="wide")
st.title("AI Video MVP Face + Line Crossing")

# ================= RTSP SETTINGS =================
st.sidebar.header("Camera Settings")

rtsp_url = st.sidebar.text_input(
    "RTSP URL",
    value="rtsp://admin:Shiv%401711@192.168.137.139:554/stream1"
)

use_rtsp = st.sidebar.checkbox("Use RTSP Camera", value=True)
# =================================================

# ---------------- SESSION STATE (IMPORTANT) ----------------
if "detector" not in st.session_state:
    st.session_state.detector = YOLODetector(MODEL_PATH)

if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker()

if "cache" not in st.session_state:
    st.session_state.cache = IdentityCache(confirm_frames=5)

if "prev_centroids" not in st.session_state:
    st.session_state.prev_centroids = {}

if "alerts" not in st.session_state:
    st.session_state.alerts = []
# ----------------------------------------------------------

# ================= CAMERA INIT =================
if use_rtsp and rtsp_url.strip() != "":
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
else:
    cap = cv2.VideoCapture(0)

# RTSP tuning (latency fix)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)
# =================================================

frame_box = st.empty()
alert_box = st.empty()

prev_time = time.time()

while True:
    # Drop old frames to reduce RTSP lag
    if use_rtsp:
        for _ in range(2):
            cap.grab()

    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Waiting for camera stream...")
        time.sleep(1)
        continue

    frame = cv2.resize(frame, (640, 480))
    detections = st.session_state.detector.detect(frame)

    boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, cls in detections]
    objects = st.session_state.tracker.update(boxes)

    for obj_id, (cx, cy) in objects.items():
        x1, y1, x2, y2 = boxes[list(objects.keys()).index(obj_id)]
        face_crop = frame[y1:y2, x1:x2]

        # ---------- FACE RECOGNITION (LOCKED) ----------
        if not st.session_state.cache.is_locked(obj_id):
            name = recognize(face_crop)
            name = st.session_state.cache.update(obj_id, name)
        else:
            name = st.session_state.cache.get_name(obj_id)
        # ------------------------------------------------

        # ---------- LINE CROSSING ----------
        if obj_id in st.session_state.prev_centroids:
            py = st.session_state.prev_centroids[obj_id][1]
            direction = crossed_line(py, cy, 200)

            if direction:
                if direction == "UP_TO_DOWN":
                    msg = f"{name} crossed the line (UP → DOWN)"
                else:
                    msg = f"{name} crossed the line (DOWN → UP)"

                if msg not in st.session_state.alerts:
                    st.session_state.alerts.append(msg)
        # ------------------------------------------------

        st.session_state.prev_centroids[obj_id] = (cx, cy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Draw line
    cv2.line(frame, (0, 200), (640, 200), (0,0,255), 2)

    # FPS
    fps = int(1 / (time.time() - prev_time))
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    alert_box.write(st.session_state.alerts[-5:])

cap.release()