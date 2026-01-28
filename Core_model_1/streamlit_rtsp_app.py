
import streamlit as st
import cv2
from ultralytics import YOLO

st.title("AI Camera Monitoring System")

# User input RTSP link
rtsp_url = st.text_input("Enter RTSP Camera Link:")

# Load model
model = YOLO("Core_Model_1.pt")

start = st.button("Start Camera")

if start and rtsp_url:
    cap = cv2.VideoCapture(rtsp_url)

    stframe = st.empty()  # Streamlit frame holder

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to connect camera")
            break

        # YOLO Detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Show in browser
        stframe.image(frame_rgb, channels="RGB")