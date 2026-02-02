


import cv2
import streamlit as st
import time

from core.detector import YOLODetector
from core.tracker import CentroidTracker
from core.face_recognizer import recognize
from core.cache import IdentityCache
from core.line_logic import crossed_line

st.set_page_config(layout="wide")
st.title("AI Video MVP – Face + Line Crossing")

cap = cv2.VideoCapture(0)

detector = YOLODetector("models/Core_Model_1.pt")
tracker = CentroidTracker()
cache = IdentityCache(confirm_frames=5)

line_y = 200
prev_centroids = {}
alerts = []

frame_box = st.empty()
alert_box = st.empty()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    detections = detector.detect(frame)

    boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, cls in detections]
    objects = tracker.update(boxes)

    for obj_id, (cx, cy) in objects.items():
        x1, y1, x2, y2 = boxes[list(objects.keys()).index(obj_id)]
        face_crop = frame[y1:y2, x1:x2]

        if not cache.is_locked(obj_id):
            name = recognize(face_crop)
            name = cache.update(obj_id, name)
        else:
            name = cache.get_name(obj_id)

        if obj_id in prev_centroids:
            py = prev_centroids[obj_id][1]

            direction = crossed_line(py, cy, line_y)

            if direction:
                if direction == "UP_TO_DOWN":
                    msg = f"{name} crossed the line (UP → DOWN)"
                else:
                    msg = f"{name} crossed the line (DOWN → UP)"

                if msg not in alerts:
                    alerts.append(msg)

            

        prev_centroids[obj_id] = (cx, cy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.line(frame, (0, line_y), (640, line_y), (0,0,255), 2)

    fps = int(1 / (time.time() - prev_time))
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    alert_box.write(alerts[-5:])

cap.release()