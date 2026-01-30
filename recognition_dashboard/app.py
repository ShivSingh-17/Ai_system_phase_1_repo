


import streamlit as st
import cv2
import time
from recognition_fast import recognize_face
from deepface import DeepFace

st.set_page_config(page_title="FAST Face Recognition", layout="wide")
st.title("AI Face Recognition Dashboard")

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame_small = cv2.resize(frame, (320, 240))

    faces = DeepFace.extract_faces(
        img_path=frame_small,
        detector_backend="opencv",
        enforce_detection=False
    )

    for face in faces:
        fa = face["facial_area"]
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

        face_crop = frame_small[y:y+h, x:x+w]

        name = recognize_face(face_crop)

        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(frame_small, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame_small, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS
    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame_small, f"FPS: {int(fps)}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb)

cap.release()


