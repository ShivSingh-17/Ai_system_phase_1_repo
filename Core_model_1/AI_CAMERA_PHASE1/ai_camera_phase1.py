

from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load model
model = YOLO("Core_Model_1.pt")

cap = cv2.VideoCapture(0)

# FPS
prev_time = 0

# Crowd density grid
GRID_SIZE = 4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLO TRACKING (IMPORTANT)
    results = model.track(frame, persist=True, conf=0.4)

    boxes = results[0].boxes
    people_count = 0

    # Crowd density grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Assume class 0 = person
            if cls == 0:
                people_count += 1

                # Draw persistent box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Crowd density grid calculation
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)
                gx = int(cx / (w/GRID_SIZE))
                gy = int(cy / (h/GRID_SIZE))
                grid[min(gy,GRID_SIZE-1)][min(gx,GRID_SIZE-1)] += 1

    # ================= CROWD DENSITY =================
    density = people_count / (h * w)

    # ================= FPS =================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # ================= STATS PANEL =================
    cv2.putText(frame, f"People Count: {people_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Crowd Density: {density:.8f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Show grid (optional debug)
    cell_w = int(w/GRID_SIZE)
    cell_h = int(h/GRID_SIZE)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cv2.rectangle(frame, (j*cell_w, i*cell_h),
                          ((j+1)*cell_w, (i+1)*cell_h), (255,255,255), 1)

    cv2.imshow("AI CAMERA PHASE 1 - TRACKING", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
