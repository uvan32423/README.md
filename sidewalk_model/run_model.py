from ultralytics import YOLO
import cv2

model = YOLO("best.onnx")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try changing 0 to 1.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read camera frame")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Sidewalk Object Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

