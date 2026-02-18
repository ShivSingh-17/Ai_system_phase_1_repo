


# core/tracker.py

import numpy as np

class Tracker:

    def __init__(self):

        self.next_id = 0
        self.objects = {}

    def _get_centroid(self, bbox):

        x1, y1, x2, y2 = bbox
        return int((x1+x2)/2), int((y1+y2)/2)

    def update(self, detections):

        tracks = []

        for det in detections:

            centroid = self._get_centroid(det["bbox"])

            track_id = self.next_id
            self.objects[track_id] = centroid
            self.next_id += 1

            tracks.append({
                "id": track_id,
                "bbox": det["bbox"]
            })

        return tracks

