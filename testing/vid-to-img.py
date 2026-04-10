import cv2
import os

video_path = "test.mp4"   # your video file
output_folder = "frames"

# create folder if not exists
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
print(f"Saved {frame_count} frames")