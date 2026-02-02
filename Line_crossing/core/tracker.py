


# core/tracker.py
import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}  # id -> centroid
        self.max_distance = max_distance

    def update(self, boxes):
        new_objects = {}
        used_ids = set()

        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            min_dist = float("inf")
            matched_id = None

            for obj_id, (px, py) in self.objects.items():
                if obj_id in used_ids:
                    continue
                dist = np.linalg.norm([cx - px, cy - py])
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                new_objects[matched_id] = (cx, cy)
                used_ids.add(matched_id)
            else:
                new_objects[self.next_id] = (cx, cy)
                used_ids.add(self.next_id)
                self.next_id += 1

        self.objects = new_objects
        return self.objects