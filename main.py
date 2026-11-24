import cv2
from ultralytics import YOLO
import config

model = YOLO(config.MODEL_NAME)

if config.SOURCE == "camera":
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
elif config.SOURCE == "video":
    cap = cv2.VideoCapture(config.VIDEO_PATH)
else:
    raise ValueError("SOURCE must be 'camera' or 'video'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()