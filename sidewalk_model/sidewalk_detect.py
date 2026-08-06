from ultralytics import YOLO
import cv2
import time
from pathlib import Path

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "best.onnx"      # Your ONNX model file
CAMERA_INDEX = 0              # Try 1 if camera 0 does not work
CONFIDENCE = 0.25             # Lower to 0.10 if it says "nothing" too often
IMAGE_SIZE = 640
SAVE_PREVIEW = True
SHOW_WINDOW = False           # Set True only if running on Jetson desktop with a monitor

# -----------------------------
# Setup
# -----------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading model...")
model = YOLO(MODEL_PATH)

print(f"Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_INDEX}. "
        "Try changing CAMERA_INDEX to 1."
    )

print("Sidewalk detection is running.")
print("Press Ctrl+C to stop.")
print("Latest preview image will be saved to outputs/latest.jpg")

last_preview_save = 0

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame.")
            time.sleep(1)
            continue

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False,
        )

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detections.append(f"{class_name} {confidence:.2f}")

        if detections:
            print(", ".join(detections), "detected")
        else:
            print("No objects detected")

        annotated_frame = results[0].plot()

        # Save one preview image per second
        now = time.time()
        if SAVE_PREVIEW and now - last_preview_save >= 1:
            cv2.imwrite(str(OUTPUT_DIR / "latest.jpg"), annotated_frame)
            last_preview_save = now

        # Optional live window, only works with a display
        if SHOW_WINDOW:
            cv2.imshow("Sidewalk Object Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nterminating program")

finally:
    cap.release()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print("program end.")

# TO RUN, ENTER 
# cd ~/sidewalk_model python3 sidewalk_detect.py
