import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("video.mp4")
model = YOLO("yolov8n.pt")
names = model.names
car_id = {}
aspect_ratio_id = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    cls = results[0].boxes.cls
    confs = results[0].boxes.conf
    for box, id, name,conf in zip(boxes, ids,cls, confs):
        if names[int(name)] == "car" or names[int(name)] == "truck":
            aspect_ratio = (box[2] - box[0]) / (box[3] - box[1])
            if aspect_ratio > 1.15 and aspect_ratio < 2.5 and conf > 0.7:
                ar = aspect_ratio_id.get(id, 0)
                if aspect_ratio > ar:
                    car_id[id] = frame[abs(box[1]):abs(box[3]), abs(box[0]):abs(box[2])]
                    aspect_ratio_id[id] = aspect_ratio


for key in car_id:
    print("id === ", key)
    cv2.imshow("",car_id[key])
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

"""from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # load an official detection model

cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    # Check if the end of the video has been reached
    if not ret:
        break

    results = model.track(source=frame, show=True,save = True, tracker="bytetrack.yaml")

cap.release()
cv2.destroyAllWindows()"""

"""cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Id {id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"name {names[int(name)]}",
            (box[0], box[3] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )"""