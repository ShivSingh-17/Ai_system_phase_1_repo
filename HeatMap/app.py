


import cv2
import streamlit as st
import time
import os

from core.detector import YOLODetector
from core.tracker import CentroidTracker
from core.heatmap_engine import HeatmapEngine
from core.utils import overlay_heatmap

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Core_Model_1.pt")
# ------------------------------------------

st.set_page_config(layout="wide")
st.title("AI Video Heatmap – Full Body Movement")

# ================= RTSP SETTINGS =================
st.sidebar.header("Camera Settings")

rtsp_url = st.sidebar.text_input(
    "RTSP URL",
    value="rtsp://username:password@192.168.1.12:554/stream1"
)

use_rtsp = st.sidebar.checkbox("Use RTSP Camera", value=True)

update_every_n = st.sidebar.slider("Heatmap update interval (frames)", 1, 5, 2)
alpha = st.sidebar.slider("Heatmap overlay intensity", 0.2, 0.7, 0.45)

if st.sidebar.button("🔄 Reset Heatmap"):
    if "heatmap" in st.session_state and st.session_state.heatmap:
        st.session_state.heatmap.reset()
# =================================================

# ---------------- SESSION STATE ----------------
if "detector" not in st.session_state:
    st.session_state.detector = YOLODetector(MODEL_PATH, conf=0.4)

if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker(max_distance=60)

if "heatmap" not in st.session_state:
    st.session_state.heatmap = None

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
# ------------------------------------------------

# ================= CAMERA INIT =================
if use_rtsp and rtsp_url.strip():
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)
# =================================================

frame_box = st.empty()
prev_time = time.time()

while True:
    # Drop old RTSP frames
    if use_rtsp:
        for _ in range(2):
            cap.grab()

    ret, frame = cap.read()
    if not ret:
        st.warning("⚠️ Waiting for camera stream...")
        time.sleep(1)
        continue

    frame = cv2.resize(frame, (640, 480))
    h, w = frame.shape[:2]

    # Init heatmap once
    if st.session_state.heatmap is None:
        st.session_state.heatmap = HeatmapEngine(w, h)

    # Detect persons
    boxes = st.session_state.detector.detect(frame)

    # Track (for stability, not for heatmap logic)
    st.session_state.tracker.update(boxes)

    # Update heatmap with FULL BODY boxes
    st.session_state.frame_count += 1
    if st.session_state.frame_count % update_every_n == 0:
        st.session_state.heatmap.update_bbox(boxes, intensity=0.3)

    # Render & overlay heatmap
    heatmap_img = st.session_state.heatmap.render()
    final_frame = overlay_heatmap(frame, heatmap_img, alpha=alpha)

    # FPS display
    fps = int(1 / max(1e-6, (time.time() - prev_time)))
    prev_time = time.time()
    cv2.putText(final_frame, f"FPS: {fps}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    frame_box.image(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))

cap.release()