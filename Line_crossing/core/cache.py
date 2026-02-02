


# core/cache.py
from collections import defaultdict, Counter

class IdentityCache:
    def __init__(self, confirm_frames=5):
        self.locked_names = {}          # track_id -> name
        self.votes = defaultdict(list)
        self.confirm_frames = confirm_frames

    def is_locked(self, track_id):
        return track_id in self.locked_names

    def get_name(self, track_id):
        return self.locked_names.get(track_id, "Detecting...")

    def update(self, track_id, predicted_name):
        if track_id in self.locked_names:
            return self.locked_names[track_id]

        self.votes[track_id].append(predicted_name)

        if len(self.votes[track_id]) >= self.confirm_frames:
            final = Counter(self.votes[track_id]).most_common(1)[0][0]
            self.locked_names[track_id] = final
            del self.votes[track_id]
            return final

        return "Detecting..."

    def remove(self, track_id):
        self.locked_names.pop(track_id, None)
        self.votes.pop(track_id, None)