import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image
import os
import time

# Video input path - update with your video file
video_path = "count2.mp4"  # Change to: one_at_time.mp4, or video/old/...

# Optional: Crop region (x1, y1, x2, y2) - set to None to disable cropping
# crop_region = (100, 100, 600, 500)  # Example: crops to region from (100,100) to (600,500)
crop_region = None  # Disable by default, set values to enable

# Optional: Start from specific timestamp in mm:ss format
start_timestamp = None  # Set to "01:30" to start from 1 min 30 sec, or None to start from beginning

# Check if video exists, if not list available videos
if not os.path.exists(video_path):
    print(f"Video not found at {video_path}")
    print("Available videos in video/new/:")
    if os.path.exists("video/new"):
        print(os.listdir("video/new"))
    exit(1)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

print(f"Video: {video_path}")
print(f"FPS: {fps}, Resolution: {frame_width}x{frame_height}, Total Frames: {total_frames}")
print(f"Duration: {int(video_duration//60):02d}:{int(video_duration%60):02d}")
if crop_region:
    print(f"Crop Region: {crop_region}")

# Seek to timestamp if specified
if start_timestamp:
    parts = start_timestamp.split(":")
    minutes = int(parts[0])
    seconds = int(parts[1])
    seek_frame = int((minutes * 60 + seconds) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
    print(f"Seeking to {start_timestamp}...")

# Optional: Setup video writer to save output
out = cv2.VideoWriter(
    "output_no_bg.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

import onnxruntime as ort
providers = ort.get_available_providers()
provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
print(f"[rembg] Using provider: {provider}")
session = new_session("u2netp", providers=[provider])

frame_count = 0
current_frame_index = 0
start_time = time.time()

print("\nProcessing frames... (press Ctrl+C to stop)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Apply crop if specified
    if crop_region:
        x1, y1, x2, y2 = crop_region
        frame = frame[y1:y2, x1:x2]

    # Resize to 320x320 for fast processing
    small_frame = cv2.resize(frame, (320, 320))
    # Convert BGR to RGB for PIL
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # # Convert BGR to RGB for PIL
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # pil_image = Image.fromarray(rgb_frame)
    
    # Remove background
    output_image = remove(pil_image, session=session)

    output_array = np.array(output_image)
    output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGR)

    # Resize back to original frame size
    output_bgr = cv2.resize(output_bgr, (frame_width, frame_height))
    
    # # Convert back to OpenCV format (BGR)
    # output_array = np.array(output_image)
    # output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGR)
    
    # Write to output video
    out.write(output_bgr)
    
    # Calculate and display FPS
    elapsed = time.time() - start_time
    current_fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Get current timestamp
    current_seconds = current_frame_index / fps
    minutes = int(current_seconds // 60)
    seconds = int(current_seconds % 60)
    
    # Show progress
    progress = (current_frame_index / total_frames) * 100
    print(f"Progress: {current_frame_index}/{total_frames} ({progress:.1f}%) | FPS: {current_fps:.1f}", end='\r')

print(f"\n✓ Processing complete! {frame_count} frames processed")
print(f"✓ Output saved to: output_no_bg.mp4")

cap.release()
out.release()
