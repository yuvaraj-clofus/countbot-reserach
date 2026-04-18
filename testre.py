import cv2
import numpy as np
from rembg import remove
from PIL import Image
import os
import time

# Video input path - update with your video file
video_path = "videos/test.mp4"  # Change to: one_at_time.mp4, or video/old/...

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

frame_count = 0
current_frame_index = 0
start_time = time.time()

print("\nProcessing frames...")
print("Controls: 'q' = quit, 'p' = pause/resume, 'SPACE' = frame step (when paused)")

paused = False

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Apply crop if specified
    if crop_region:
        x1, y1, x2, y2 = crop_region
        frame = frame[y1:y2, x1:x2]
    
    # Convert BGR to RGB for PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Remove background
    output_image = remove(pil_image)
    
    # Convert back to OpenCV format (BGR)
    output_array = np.array(output_image)
    output_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGR)
    
    # Write to output video
    out.write(output_bgr)
    
    # Calculate and display FPS
    elapsed = time.time() - start_time
    current_fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Get current timestamp
    current_seconds = current_frame_index / fps
    minutes = int(current_seconds // 60)
    seconds = int(current_seconds % 60)
    
    # Add FPS and timestamp to frame
    display_frame = cv2.resize(output_bgr, (960, 540))
    status_text = f"FPS: {current_fps:.1f} | Time: {minutes:02d}:{seconds:02d} | Frame: {current_frame_index}/{total_frames}"
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if paused:
        cv2.putText(display_frame, "PAUSED (Press SPACE to step, P to resume)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Background Removed - Press Q to exit", display_frame)
    
    # Show progress
    progress = (current_frame_index / total_frames) * 100
    print(f"Progress: {current_frame_index}/{total_frames} ({progress:.1f}%) | FPS: {current_fps:.1f}", end='\r')
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n✓ Quit requested")
        break
    elif key == ord('p'):
        paused = not paused
        start_time = time.time() - (frame_count / fps)
        print(f"\n{'Paused' if paused else 'Resumed'}", end='\r')
    elif key == ord(' ') and paused:
        # Step one frame forward when paused
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

print(f"\n✓ Processing complete! {frame_count} frames processed")
print(f"✓ Output saved to: output_no_bg.mp4")

cap.release()
out.release()
cv2.destroyAllWindows()
