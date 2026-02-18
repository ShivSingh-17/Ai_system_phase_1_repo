


# app.py

import streamlit as st
import cv2
import time

from core.detector import PPEDetector
from core.logic import PPELogic

# ---------------- CONFIG ---------------- #

MODEL_PATH = "models/ppe_best.pt"

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots"
}

# ---------------------------------------- #

st.set_page_config(layout="wide")
st.title("🦺 PPE Detection Dashboard")

video_path = st.text_input("Enter RTSP / Video Path")

run = st.button("Start Monitoring")

frame_window = st.empty()

if run and video_path:

    # 🔧 RTSP buffer optimization
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    detector = PPEDetector(MODEL_PATH)
    logic = PPELogic()

    frame_count = 0
    SKIP_FRAMES = 2   # 🔥 FPS boost

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # 🔧 Resize frame (huge FPS gain)
        frame = cv2.resize(frame, (640, 480))

        frame_count += 1

        # 🔧 Skip frames
        if frame_count % SKIP_FRAMES != 0:
            frame_window.image(frame, channels="BGR")
            continue

        detections = detector.detect(frame)

        detected_items = []

        for det in detections:

            cls_name = CLASS_NAMES[det["class_id"]]
            detected_items.append(cls_name)

            x1, y1, x2, y2 = map(int, det["bbox"])

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        cls_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        # PPE Logic Check
        status, missing = logic.update(0, detected_items)

        if status is False:
            cv2.putText(frame,
                        f"ALERT Missing: {missing}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        frame_window.image(frame, channels="BGR")

        # 🔧 Small delay for UI smoothness
        time.sleep(0.01)

    cap.release()







