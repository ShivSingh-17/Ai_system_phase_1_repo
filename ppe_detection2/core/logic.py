


# core/ppe_logic.py

import time


class PPELogic:

    def __init__(self, alert_delay=5):

        # Required PPE items
        self.required_items = {
            "helmet",
            "gloves",
            "vest",
            "boots"
        }

        # Track person PPE history
        self.track_history = {}

        # Alert delay (seconds)
        self.alert_delay = alert_delay

    # ------------------------------------------------ #

    def update(self, track_id, detected_items):

        """
        track_id : person id
        detected_items : list of detected PPE items
        """

        current_time = time.time()

        detected_set = set(detected_items)

        missing_items = self.required_items - detected_set

        # First time seeing this person
        if track_id not in self.track_history:

            self.track_history[track_id] = {
                "first_seen": current_time,
                "last_missing": missing_items
            }

            return True, []

        history = self.track_history[track_id]

        # If PPE complete
        if len(missing_items) == 0:

            # Reset timer
            history["first_seen"] = current_time
            history["last_missing"] = set()

            return True, []

        # PPE Missing → check delay
        time_elapsed = current_time - history["first_seen"]

        if time_elapsed >= self.alert_delay:

            return False, list(missing_items)

        else:

            return True, list(missing_items)



