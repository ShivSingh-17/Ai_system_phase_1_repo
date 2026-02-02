


import cv2
import streamlit as st
import time
import os

from core.detector import YOLODetector
from core.tracker import CentroidTracker
from core.face_recognizer import recognize
from core.presence_cache import PresenceManager

# ---------------- PATH FIX ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "Core_Model_1.pt")
# ------------------------------------------

st.set_page_config(layout="wide")
st.title("AI Video Entry / Exit Dashboard")

# ---------------- SESSION STATE ----------------
if "detector" not in st.session_state:
    st.session_state.detector = YOLODetector(MODEL_PATH)

if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker()

if "presence" not in st.session_state:
    st.session_state.presence = PresenceManager(confirm_frames=3, exit_delay=60)

if "last_recog_time" not in st.session_state:
    st.session_state.last_recog_time = {}
# ------------------------------------------------

cap = cv2.VideoCapture(0)

frame_box = st.empty()
col1, col2 = st.columns(2)

entry_placeholder = col1.empty()
exit_placeholder = col2.empty()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    boxes = st.session_state.detector.detect(frame)
    objects = st.session_state.tracker.update(boxes)

    for obj_id, (cx, cy) in objects.items():
        x1, y1, x2, y2 = boxes[list(objects.keys()).index(obj_id)]
        face_crop = frame[y1:y2, x1:x2]

        # ---------- FACE RECOGNITION (THROTTLED) ----------
        if obj_id not in st.session_state.presence.locked:
            now = time.time()
            last_time = st.session_state.last_recog_time.get(obj_id, 0)

            if now - last_time > 1.0:
                predicted = recognize(face_crop)
                name = st.session_state.presence.update_identity(obj_id, predicted)
                st.session_state.last_recog_time[obj_id] = now
            else:
                name = "Detecting..."
        else:
            name = st.session_state.presence.locked[obj_id]
            st.session_state.presence.seen(obj_id)
        # --------------------------------------------------

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    st.session_state.presence.check_exit()

    frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    entry_placeholder.markdown("### 🟢 Entry Logs")
    entry_placeholder.write(st.session_state.presence.entry_logs[-10:])

    exit_placeholder.markdown("### 🔴 Exit Logs")
    exit_placeholder.write(st.session_state.presence.exit_logs[-10:])

cap.release()