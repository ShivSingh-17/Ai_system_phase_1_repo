


# core/ppe_detector.py

from ultralytics import YOLO
import cv2


class PPEDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        # PPE classes mapping
        self.class_names = {
            0: "helmet",
            1: "gloves",
            2: "vest",
            3: "boots"
        }

    def detect(self, frame):

        results = self.model(frame, conf=0.4, iou=0.5)[0]

        detections = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, score, cid in zip(boxes, scores, class_ids):

            if cid not in self.class_names:
                continue

            x1, y1, x2, y2 = map(int, box)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(score),
                "class_id": cid,
                "label": self.class_names[cid]
            })

        return detections
