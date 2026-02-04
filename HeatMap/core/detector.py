


from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Returns list of bounding boxes for persons:
        [(x1, y1, x2, y2), ...]
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])

            # YOLO person class = 0
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))

        return boxes