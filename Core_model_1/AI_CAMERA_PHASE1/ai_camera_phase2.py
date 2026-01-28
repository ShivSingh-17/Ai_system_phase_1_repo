
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load model
model = YOLO("Core_Model_1.pt")
cap = cv2.VideoCapture(0)

# Heatmap accumulator
heatmap = None
decay = 0.95   # heatmap fade factor

# Virtual queue line
QUEUE_LINE_Y = 300
queue_count = 0
tracked_ids = set()

# FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # Run detection
    results = model(frame, conf=0.4)
    boxes = results[0].boxes

    people_count = 0
    centers = []

    for box in boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Only count humans (change class id if needed)
        if cls == 0:
            people_count += 1
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))

            # Heatmap accumulation
            heatmap[cy, cx] += 1

            # Queue detection
            if cy > QUEUE_LINE_Y and cx not in tracked_ids:
                queue_count += 1
                tracked_ids.add(cx)

    # ================= CROWD DENSITY =================
    area = h * w
    density = people_count / area

    # ================= HEATMAP VISUAL =================
    heatmap = heatmap * decay
    heatmap_color = cv2.applyColorMap(
        np.uint8(np.clip(heatmap, 0, 255)), cv2.COLORMAP_JET
    )
    heatmap_overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # ================= QUEUE LINE =================
    cv2.line(heatmap_overlay, (0, QUEUE_LINE_Y), (w, QUEUE_LINE_Y), (0,255,255), 2)

    # ================= FPS =================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # ================= STATS PANEL =================
    cv2.putText(heatmap_overlay, f"People: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(heatmap_overlay, f"Density: {density:.6f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(heatmap_overlay, f"Queue Count: {queue_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(heatmap_overlay, f"FPS: {int(fps)}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    cv2.imshow("AI CAMERA PHASE 2 - Crowd & Queue", heatmap_overlay)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
