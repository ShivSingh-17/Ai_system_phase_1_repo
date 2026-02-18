


# core/ppe_logic.py

import time

class PPELogic:

    REQUIRED = ["helmet", "gloves", "vest", "boots"]

    def __init__(self):

        self.history = {}

    def update(self, track_id, detected_items):

        now = time.time()

        if track_id not in self.history:
            self.history[track_id] = now

        missing = [
            item for item in self.REQUIRED
            if item not in detected_items
        ]

        if not missing:
            self.history[track_id] = now
            return True, []

        elapsed = now - self.history[track_id]

        if elapsed >= 5:
            return False, missing

        return True, []



