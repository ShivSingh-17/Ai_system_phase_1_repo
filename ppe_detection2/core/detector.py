


# core/ppe_detector.py

from ultralytics import YOLO


class PPEDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, frame):

        # 🔧 Lower imgsz for faster inference
        results = self.model(
            frame,
            imgsz=480,
            conf=0.4,
            device=0,
            verbose=False
        )[0]

        detections = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = box

            detections.append({
                "class_id": int(cls),
                "bbox": [x1, y1, x2, y2]
            })

        return detections

