


import cv2

def overlay_heatmap(frame, heatmap, alpha=0.4):
    """
    Overlay heatmap on frame
    """
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)