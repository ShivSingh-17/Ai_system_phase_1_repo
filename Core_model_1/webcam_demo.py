from ultralytics import YOLO
import cv2

model = YOLO("Core_Model_1.pt")          # Loading our trained model her

#Core_Model_1 is trained on kaggle dataset(medium size images) for better performance for initial phase 

cap = cv2.VideoCapture(0)     # Open the webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    cv2.imshow("AI Camera Core Model", annotated)       # foor output

    if cv2.waitKey(1) == 27:  #     exit
        break

cap.release()
cv2.destroyAllWindows()
