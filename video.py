import cv2
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

USE_GSTREAMER = False

if USE_GSTREAMER:
    pipeline = (
        "v4l2src device=/dev/video1 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

# Get resolution (IMPORTANT: swap for 90° rotation)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------- OUTPUT FILE --------
output_file = "count2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Swap width & height because of rotation
out = cv2.VideoWriter(output_file, fourcc, 20.0, (height, width))

print(f"✅ Recording started. Saving to: {output_file}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔄 Rotate 90° clockwise (vertical → horizontal)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Save rotated frame
    out.write(frame)

    cv2.imshow("Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved successfully!")