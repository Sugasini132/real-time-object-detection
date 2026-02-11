from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")  # better accuracy

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    results = model(
        frame,
        conf=0.25,
        iou=0.45
    )

    for r in results:
        frame = r.plot()

    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

