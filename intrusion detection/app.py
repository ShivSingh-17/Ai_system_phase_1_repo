


import cv2
import streamlit as st
import os
import time
import threading

from core.detector import YOLODetector
from core.tracker import CentroidTracker
from core.face_recognizer import recognize
from core.identity_cache import IdentityCache
from core.intrusion_logic import IntrusionDetector

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Core_Model_1.pt")

AUTHORIZED = ["shiv", "amit", "rahul"]

# ---------------- SESSION ----------------
if "detector" not in st.session_state:
    st.session_state.detector = YOLODetector(MODEL_PATH)

if "tracker" not in st.session_state:
    st.session_state.tracker = CentroidTracker()

if "cache" not in st.session_state:
    st.session_state.cache = IdentityCache()

if "intrusion" not in st.session_state:
    st.session_state.intrusion = IntrusionDetector(AUTHORIZED)

if "last_recog_time" not in st.session_state:
    st.session_state.last_recog_time = {}

# 🔥 GLOBAL TIMER
if "global_presence_start" not in st.session_state:
    st.session_state.global_presence_start = None
# ------------------------------------------------

st.set_page_config(layout="wide")
st.title("AI Video Intrusion Detection")

rtsp_url = st.sidebar.text_input(
    "RTSP URL",
    value="rtsp://username:password@192.168.1.12:554/stream1"
)

# ---------------- RTSP THREAD ----------------
class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False

        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame

stream = RTSPStream(rtsp_url)

frame_box = st.empty()
alert_box = st.empty()

prev_time = time.time()

while True:

    ret, frame = stream.read()

    if not ret or frame is None:
        st.warning("Waiting for stream...")
        time.sleep(0.5)
        continue

    frame = cv2.resize(frame, (640, 480))

    boxes = st.session_state.detector.detect(frame)
    objects = st.session_state.tracker.update(boxes)

    person_present = len(objects) > 0

    # -------- GLOBAL TIMER START --------
    if person_present:

        if st.session_state.global_presence_start is None:
            st.session_state.global_presence_start = time.time()

    else:
        st.session_state.global_presence_start = None
    # -----------------------------------

    for obj_id, _ in objects.items():

        x1, y1, x2, y2 = boxes[list(objects.keys()).index(obj_id)]
        pad = 20
        h, w = frame.shape[:2]

        x1p = max(0, x1-pad)
        y1p = max(0, y1-pad)
        x2p = min(w, x2+pad)
        y2p = min(h, y2+pad)

        face_crop = frame[y1p:y2p, x1p:x2p]

        now = time.time()
        last_time = st.session_state.last_recog_time.get(obj_id, 0)

        # -------- FACE RECOG --------
        if obj_id not in st.session_state.cache.locked:

            if now - last_time > 2.0:
                predicted = recognize(face_crop)
                name = st.session_state.cache.update(obj_id, predicted)
                st.session_state.last_recog_time[obj_id] = now
            else:
                name = "Detecting..."

        else:
            name = st.session_state.cache.locked[obj_id]
        # ----------------------------

        # -------- UNAUTHORIZED --------
        if name not in ["Detecting..."]:

            if name == "Unknown" or name not in AUTHORIZED:
                st.session_state.intrusion.check_intrusion(name)

            # Reset global timer if recognized
            st.session_state.global_presence_start = None
        # --------------------------------

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # -------- GLOBAL DETECTING TIMEOUT --------
    if st.session_state.global_presence_start is not None:

        elapsed = time.time() - st.session_state.global_presence_start

        if elapsed > 5:
            st.session_state.intrusion.person_not_detected()
            st.session_state.global_presence_start = None
    # -----------------------------------------

    fps = int(1 / max(1e-6, (time.time() - prev_time)))
    prev_time = time.time()

    cv2.putText(frame, f"FPS: {fps}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    alert_box.write(st.session_state.intrusion.alerts)

    time.sleep(0.02)
