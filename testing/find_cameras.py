import cv2

def list_cameras(max_tested=10):
    print("Scanning for available cameras...\n")
    
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        
        if cap is None or not cap.isOpened():
            # Camera not available at this index
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera found at index: {index}")
        else:
            print(f"⚠️ Camera opened but no frame at index: {index}")
        
        cap.release()

if __name__ == "__main__":
    list_cameras()