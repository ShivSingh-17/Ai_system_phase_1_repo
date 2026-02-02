


import time
from collections import defaultdict, Counter

class PresenceManager:
    def __init__(self, confirm_frames=3, exit_delay=60):
        self.locked = {}                 # track_id -> name
        self.votes = defaultdict(list)

        self.present = {}                # track_id -> last_seen_time
        self.confirm_frames = confirm_frames
        self.exit_delay = exit_delay

        self.entry_logs = []
        self.exit_logs = []

    def update_identity(self, track_id, name):
        if track_id in self.locked:
            return self.locked[track_id]

        self.votes[track_id].append(name)

        if len(self.votes[track_id]) >= self.confirm_frames:
            final = Counter(self.votes[track_id]).most_common(1)[0][0]
            self.locked[track_id] = final

            now = time.strftime("%H:%M:%S")
            self.entry_logs.append(f"{final} entered at {now}")

            self.present[track_id] = time.time()
            del self.votes[track_id]

            return final

        return "Detecting..."

    def seen(self, track_id):
        self.present[track_id] = time.time()

    def check_exit(self):
        now = time.time()
        for track_id in list(self.present.keys()):
            last_seen = self.present[track_id]

            if now - last_seen >= self.exit_delay:
                name = self.locked.get(track_id, "Unknown")
                t = time.strftime("%H:%M:%S")
                self.exit_logs.append(f"{name} exited at {t}")

                self.present.pop(track_id)
                self.locked.pop(track_id, None)