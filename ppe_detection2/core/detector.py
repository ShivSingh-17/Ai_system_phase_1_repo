


from ultralytics import YOLO

class PPEDetector:

    def __init__(self, model_path):

        self.ppe_model = YOLO(model_path)
        self.person_model = YOLO("yolov8n.pt")

    def detect_persons(self, frame):

        results = self.person_model(frame, conf=0.4, verbose=False)[0]

        persons = []

        if results.boxes is None:
            return persons

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            if int(cls) == 0:  # person class

                persons.append({
                    "bbox": box
                })

        return persons

    def detect_ppe(self, frame):

        results = self.ppe_model(frame, conf=0.4, verbose=False)[0]

        detections = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            detections.append({
                "bbox": box,
                "class_id": int(cls)
            })

        return detections
