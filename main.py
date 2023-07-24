import cv2
import torch
import os
from PIL import Image
import numpy as np
from tracker import *
from ultralytics import YOLO




# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = YOLO("yolov8s.pt")

tracker = Sort()
# Images
cap = cv2.VideoCapture("video.mp4")

# Create a list to keep track of the already saved car images
saved_cars = []
ids = []
# Get the class names
class_names = model.names
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    # Check if the end of the video has been reached
    if not ret:
        break

    # Inference
    results = model(frame) # batch of images

    # Results
    #results.print()
    #results.show()  # or .show()
    detections = []
    # Loop through the detected objects
    print(results)
    for obj in results.xyxy[0]:
        #print(obj)
        # Extract the class label and confidence score
        class_label = class_names[int(obj[-1])]
        confidence_score = obj[-2]
        # Check if the object is a car with high confidence
        if (class_label == "car" or class_label == "truck")  and confidence_score > 0.7:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = [int(x) for x in obj[:-2]]
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = (x2 - x1) / (y2 - y1)
            if aspect_ratio > 1.15 and aspect_ratio < 2.5:
              detections.append([x1,y1,x2,y2])
              #cv2.imshow("",frame[y1:y2,x1:x2])
              #cv2.waitKey(1)
              #print(x1,y1,x2,y2)

    if len(detections) == 0:
      continue
    detections = np.array(detections)
    boxes_id = tracker.update(detections)
    for box_id in boxes_id:
        x1,y1,x2,y2,id = box_id
        x1 = round(abs(x1))
        y1 = round(abs(y1))
        x2 = round(abs(x2))
        y2 = round(abs(y2))
        if id in ids:
            continue
        #x2 = x+w
        #y2 = y+h
        #print(x1,y1,x2,y2)
        #cv2.imshow("",frame[y1:y2, x1:x2])
        #cv2.waitKey(1)
        car = frame[y1:y2, x1:x2]
        ids.append(id)
        saved_cars.append(car)
        filename = f"{x1}_{y1}_{x2}_{y2}.jpg"
        cv2.imwrite(os.path.join("saved_cars", filename), car)

for car in saved_cars:
  cv2.imshow("",car)
  cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

