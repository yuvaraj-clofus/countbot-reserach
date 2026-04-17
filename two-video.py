import cv2

# Open both cameras
cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)  # GS camera
cap1 = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Lenovo webcam

# (Important for webcam)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap0.isOpened() or not cap1.isOpened():
    print("❌ Error: One or both cameras not opened")
    exit()

# Get resolution
w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------- OUTPUT FILES --------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# GS cam → rotated → swap width & height
out0 = cv2.VideoWriter("cam0.mp4", fourcc, 20, (h0, w0))

# Webcam → normal
out1 = cv2.VideoWriter("cam1.mp4", fourcc, 20, (w1, h1))

print("✅ Recording started (2 cameras). Press 'q' to stop.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("❌ Frame grab failed")
        break

    # 🔄 Rotate GS camera (CORRECT WAY)
    frame0 = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)

    # Show both
    cv2.imshow("Cam 0 - GS (Rotated)", frame0)
    cv2.imshow("Cam 1 - Webcam", frame1)

    # Save both
    out0.write(frame0)
    out1.write(frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap0.release()
cap1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()

print("✅ Saved: cam0.mp4 and cam1.mp4")