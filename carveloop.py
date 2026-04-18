import cv2
import numpy as np
from PIL import Image
from carvekit.api.high import HiInterface

# interface = HiInterface(
#     object_type="object",
#     batch_size_seg=1,
#     batch_size_matting=1,
#     device="cpu",
#     seg_mask_size=480,
#     matting_mask_size=1024,
#     trimap_dilation=30,
#     trimap_erosion_iters=5,
#     fp16=False
# )

interface = HiInterface(
    object_type="object",
    batch_size_seg=1,
    batch_size_matting=1,
    device="cuda",
    seg_mask_size=320,
    matting_mask_size=512,
    trimap_dilation=15,
    trimap_erosion_iters=3,
    fp16=True
)

START = "00:00"  # mm:ss

cap = cv2.VideoCapture("count2.mp4")
print(f"Video opened: {cap.isOpened()}")
print(f"Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

m, s = (int(x) for x in START.split(":"))
cap.set(cv2.CAP_PROP_POS_MSEC, (m * 60 + s) * 1000)

fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
tick_freq = cv2.getTickFrequency()
frame_clock = cv2.getTickCount()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Stopped at frame {frame_count}")
        break
    frame_count += 1

    # Auto-skip frames to match real-time speed
    elapsed_sec = (cv2.getTickCount() - frame_clock) / tick_freq
    for _ in range(max(0, int(elapsed_sec * fps) - 1)):
        if not cap.grab():
            break
        frame_count += 1
    frame_clock = cv2.getTickCount()

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Remove background
    result = interface([pil_img])[0]  # RGBA

    # Convert back to BGR with white background
    bg = Image.new("RGB", result.size, (255, 255, 255))
    bg.paste(result, mask=result.split()[3])  # Alpha mask
    output = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

    combined = np.hstack([frame, output])
    cv2.imshow("No BG", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()