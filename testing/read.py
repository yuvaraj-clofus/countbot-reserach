import cv2
import zxingcpp

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = zxingcpp.read_barcodes(gray)

    for result in results:
        text = result.text
        print("Detected:", text)

        pos = result.position

        pts = [
            (int(pos.top_left.x), int(pos.top_left.y)),
            (int(pos.top_right.x), int(pos.top_right.y)),
            (int(pos.bottom_right.x), int(pos.bottom_right.y)),
            (int(pos.bottom_left.x), int(pos.bottom_left.y)),
        ]

        # Draw box
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i+1) % 4], (0, 255, 0), 2)

        # Put text
        cv2.putText(frame, text, pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("QR Reader", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()