


import streamlit as st
import cv2
import time

from core.detector import PPEDetector
from core.tracker import Tracker
from core.face_recognizer import FaceRecognizer
from core.identity_cache import IdentityCache
from core.identity_logic import IdentityLogic
from core.logic import PPELogic

MODEL_PATH = "models/ppe_best.pt"

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots"
}

st.set_page_config(layout="wide")
st.title("PPE + Identity Monitoring")

rtsp = st.text_input("RTSP / Video Path")
run = st.button("Start Monitoring")

frame_window = st.empty()

if run and rtsp:

    cap = cv2.VideoCapture(rtsp)

    detector = PPEDetector(MODEL_PATH)
    tracker = Tracker()

    recognizer = FaceRecognizer(
        "face_database/face_embeddings.pkl"
    )

    cache = IdentityCache()
    identity_logic = IdentityLogic(recognizer, cache)
    ppe_logic = PPELogic()

    prev_time = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # -------- FPS CONTROL -------- #
        if time.time() - prev_time < 0.1:
            continue
        prev_time = time.time()

        frame = cv2.resize(frame, (960,540))

        persons = detector.detect_persons(frame)
        ppe_items = detector.detect_ppe(frame)

        tracks = tracker.update(persons)

        for track in tracks:

            track_id = track["id"]
            x1,y1,x2,y2 = map(int, track["bbox"])

            # -------- FACE CROP -------- #
            face_crop = frame[
                y1:int(y1+(y2-y1)*0.3),
                x1:x2
            ]

            name = identity_logic.update(
                track_id,
                face_crop
            )

            detected_items = []

            # -------- PPE INSIDE PERSON -------- #
            for item in ppe_items:

                ix1,iy1,ix2,iy2 = map(int,item["bbox"])

                if ix1 > x1 and ix2 < x2 and iy1 > y1 and iy2 < y2:
                    detected_items.append(
                        CLASS_NAMES[item["class_id"]]
                    )

            status, missing = ppe_logic.update(
                track_id,
                detected_items
            )

            label = f"{name} | PPE OK"
            color = (0,255,0)

            if not status:
                label = f"{name} Missing: {missing}"
                color = (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,color,2)

        frame_window.image(frame, channels="BGR")

    cap.release()

