


# core/ppe_logic.py

import time


REQUIRED_PPE = ["helmet", "gloves", "vest", "boots"]


class PPELogic:

    def __init__(self, alert_delay=5):

        self.alert_delay = alert_delay

        # track_id → first missing time
        self.missing_timer = {}

    def check_ppe(self, track_id, detected_items):

        missing_items = [
            item for item in REQUIRED_PPE
            if item not in detected_items
        ]

        current_time = time.time()

        # If PPE complete
        if len(missing_items) == 0:

            if track_id in self.missing_timer:
                del self.missing_timer[track_id]

            return {
                "status": "complete",
                "missing": []
            }

        # PPE incomplete
        if track_id not in self.missing_timer:
            self.missing_timer[track_id] = current_time
            return None

        elapsed = current_time - self.missing_timer[track_id]

        if elapsed >= self.alert_delay:
            return {
                "status": "incomplete",
                "missing": missing_items
            }

        return None
