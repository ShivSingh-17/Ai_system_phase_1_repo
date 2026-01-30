


import streamlit as st
import cv2
import os

DB_PATH = "../face_database"
os.makedirs(DB_PATH, exist_ok=True)

st.title("Face Register Dashboard")

name = st.text_input("Enter Person Name")
capture = st.button("Register Face")

cap = cv2.VideoCapture(0)
frame_box = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        face_crop = frame[y:y+h, x:x+w]

        if capture and name != "":
            person_folder = os.path.join(DB_PATH, name)
            os.makedirs(person_folder, exist_ok=True)
            cv2.imwrite(f"{person_folder}/face.jpg", face_crop)
            st.success(f"Registered {name}")
            cap.release()
            cv2.destroyAllWindows()
            st.stop()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_box.image(frame_rgb)