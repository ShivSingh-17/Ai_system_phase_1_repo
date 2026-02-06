


import cv2
import streamlit as st
import time
import os
import threading

from core.detector import PersonDetector
from core.loitering_logic import CrowdLoitering

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Core_Model_1.pt")

# ---------------- INIT ----------------
if "detector" not in st.session_state:
    st.session_state.detector = PersonDetector(MODEL_PATH)

if "loiter" not in st.session_state:
    st.session_state.loiter = CrowdLoitering()

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("Crowd Loitering Detection Dashboard")

st.sidebar.header("Loitering Controls")

people_threshold = st.sidebar.number_input(
    "People Count Threshold",
    min_value=1,
    max_value=20,
    value=3
)

time_threshold = st.sidebar.number_input(
    "Loiter Time (seconds)",
    min_value=5,
    max_value=300,
    value=30
)

# ---------------- CAMERA MODE ----------------
camera_mode = st.sidebar.radio(
    "Camera Source",
    ["Webcam", "RTSP"]
)

rtsp_url = ""

if camera_mode == "RTSP":
    rtsp_url = st.sidebar.text_input(
        "RTSP URL",
        value="rtsp://username:password@192.168.1.12:554/stream1"
    )

# ---------------- RTSP THREAD CLASS ----------------
class RTSPStream:

    def __init__(self, url):

        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None

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

# ---------------- CAMERA INIT ----------------
if camera_mode == "Webcam":
    cap = cv2.VideoCapture(0)

else:
    stream = RTSPStream(rtsp_url)

frame_box = st.empty()
alert_box = st.empty()

# ================= MAIN LOOP =================
while True:

    # -------- Frame Read --------
    if camera_mode == "Webcam":

        ret, frame = cap.read()

    else:

        ret, frame = stream.read()

        if not ret or frame is None:
            st.warning("Waiting for RTSP stream...")
            time.sleep(0.5)
            continue

    if not ret:
        break

    frame = cv2.resize(frame, (800, 500))

    # -------- Person Detection --------
    boxes = st.session_state.detector.detect(frame)
    person_count = len(boxes)

    # -------- Draw Bounding Boxes --------
    for (x1, y1, x2, y2) in boxes:

        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      2)

        cv2.putText(frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # -------- Loiter Logic --------
    alert, duration = st.session_state.loiter.update(
        person_count,
        people_threshold,
        time_threshold
    )

    # -------- Overlay Text --------
    cv2.putText(frame,
                f"Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2)

    if alert:

        cv2.putText(frame,
                    f"LOITERING ALERT ({int(duration)}s)",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

        alert_box.error(
            f"Crowd loitering detected ({person_count} persons, {int(duration)} sec)"
        )

    else:

        alert_box.success("No loitering detected")

    # -------- Show Frame --------
    frame_box.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB"
    )