


# app.py

import streamlit as st
import cv2

from core.ppe_detector import PPEDetector
from core.ppe_logic import PPELogic
from core.tracker import Tracker
from core.face_recognizer import FaceRecognizer


MODEL_PATH = "models/ppe_best.pt"


st.title("PPE Detection + Identity Test")

source = st.text_input(
    "RTSP / Video Path / 0 for Webcam",
    "0"
)

identity_on = st.checkbox("Enable Face Recognition")

start = st.button("Start Monitoring")


if start:

    # Load modules
    detector = PPEDetector(MODEL_PATH)
    logic = PPELogic(alert_delay=5)
    tracker = Tracker()

    if identity_on:
        face_recognizer = FaceRecognizer(
            "face_database/face_embeddings.pkl"
        )

    cap = cv2.VideoCapture(
        0 if source == "0" else source
    )

    frame_placeholder = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        tracks = tracker.update(detections)

        for track in tracks:

            track_id = track["id"]
            bbox = track["bbox"]
            items = track["labels"]

            result = logic.check_ppe(track_id, items)

            x1, y1, x2, y2 = bbox

            name = "Unknown"

            if identity_on:
                name = face_recognizer.recognize(frame, bbox)

            # Draw box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"ID {track_id} | {name}"

            if result and result["status"] == "incomplete":

                missing = ",".join(result["missing"])

                label += f" ❌ Missing: {missing}"

                cv2.putText(
                    frame,
                    "ALERT!",
                    (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        frame_placeholder.image(
            frame,
            channels="BGR"
        )

    cap.release()

