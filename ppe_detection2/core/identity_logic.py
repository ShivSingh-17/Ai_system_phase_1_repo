


# core/identity_logic.py

import time

class IdentityLogic:

    def __init__(self, recognizer, cache):

        self.recognizer = recognizer
        self.cache = cache
        self.cooldown = 5
        self.last_seen = {}

    def update(self, track_id, face_crop):

        if self.cache.exists(track_id):
            return self.cache.get(track_id)

        now = time.time()

        if track_id in self.last_seen:
            if now - self.last_seen[track_id] < self.cooldown:
                return "Unknown"

        name = self.recognizer.recognize(face_crop)

        self.cache.update(track_id, name)
        self.last_seen[track_id] = now

        return name
