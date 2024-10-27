import cv2

# video_path = 'input_video.mp4'
# cap = cv2.VideoCapture(video_path)

# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imwrite(f'frames/frame_{frame_count:04d}.png', frame)
#     frame_count += 1

# cap.release()

import numpy as np

# Load frames
frame_indices = [90, 700, 325]

# Read the specified frames
frames = [cv2.imread(f'frames/frame_{i:04d}.png') for i in frame_indices]

# Create a blank canvas
height, width, _ = frames[0].shape
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Overlay frames on the canvas with transparency
alpha = 0.55
for frame in frames:
    cv2.addWeighted(frame, alpha, canvas, 1 - alpha, 0, canvas)

# Save the composite image
cv2.imwrite('handover_movement.png', canvas)