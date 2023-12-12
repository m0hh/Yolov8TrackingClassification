import cv2
from ultralytics import YOLO
from fastai.vision.all import *
import pandas as pd



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


learner = load_learner("Model3")
final_predictions = {}
for key in car_id:
    prediction, _, _ = learner.predict(car_id[key])
    final_predictions[prediction] = final_predictions.get(prediction,0) + 1


cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame([final_predictions])
df.to_csv("predictions.csv")
