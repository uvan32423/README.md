from ultralytics import YOLO

MODEL_PATH = "best.onnx"
IMAGE_PATH = "test2.jpg"
CONFIDENCE = 0.25
IMAGE_SIZE = 640

model = YOLO(MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    save=True,
    verbose=False,
)

found = False

for result in results:
    for box in result.boxes:
        found = True
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"{class_name} detected: {confidence:.2f}")

if not found:
    print("No objects detected. Try lowering CONFIDENCE to 0.10.")
