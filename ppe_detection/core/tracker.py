


import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=60):
        self.next_id = 0
        self.objects = {}

    def update(self, boxes):
        new_objects = {}

        for (x1, y1, x2, y2) in boxes:
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            new_objects[self.next_id] = (cx, cy)
            self.next_id += 1

        self.objects = new_objects
        return self.objects