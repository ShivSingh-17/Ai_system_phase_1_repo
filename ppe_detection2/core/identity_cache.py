


# core/identity_cache.py

class IdentityCache:

    def __init__(self):

        self.cache = {}

    def update(self, track_id, name):

        self.cache[track_id] = name

    def get(self, track_id):

        return self.cache.get(track_id, "Unknown")

    def exists(self, track_id):

        return track_id in self.cache
