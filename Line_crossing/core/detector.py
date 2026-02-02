


# core/detector.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, conf=0.4, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, cls))

        return detections