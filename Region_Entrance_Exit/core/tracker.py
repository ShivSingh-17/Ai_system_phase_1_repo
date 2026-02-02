


import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=60):
        self.next_id = 0
        self.objects = {}

    def update(self, boxes):
        new_objects = {}
        used = set()

        for (x1, y1, x2, y2) in boxes:
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            best_id, min_dist = None, 1e9
            for oid, (px, py) in self.objects.items():
                if oid in used:
                    continue
                d = np.linalg.norm([cx-px, cy-py])
                if d < min_dist and d < 60:
                    min_dist, best_id = d, oid

            if best_id is not None:
                new_objects[best_id] = (cx, cy)
                used.add(best_id)
            else:
                new_objects[self.next_id] = (cx, cy)
                used.add(self.next_id)
                self.next_id += 1

        self.objects = new_objects
        return self.objects