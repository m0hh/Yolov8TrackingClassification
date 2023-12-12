# Identifying car models from a video using YOLO for car identification and tracking and resnet50 for classification

## How the program works
This program takes a video of filmed cars and outputs a csv file with car models and makes and how many times each model-make is identified
## Part 1 car identification and tracking

File: yolo.py
We use YOLOV8 for car identification and tracking.
First we car an empty car_id dict and an empty aspect_ratio_id dict, we then read the frame and exctract the boxes, ids,names and confidance scores.
We zip them together and loop over each one.
If the name is a car or a truck we calculate the ascpect ration of the image exctracted by the boxes from the frame, we then check if the aspect ratio is bigger than 1.15 and smaller than 2.5, this helps us to only select frames where a whole car is present in the image because sometime yolo will identify a car while it's not yet fully visible, we also check if the confidence score is higher than 0.7, this helps us to weed out any blurry exctracted images because if the videos are filmed by hand a lot of frames will have blurry images.
We then check by the object id our aspect_ratio_id dict and see what aspect ratio is currently saved for that id and check it against the current aspect ration of the current image we are inspecting, if the aspect ration of the current image is bigger than the aspect ratio currently saved for that id in our aspect_ratio_id dict we update our aspect_ratio_id with the new aspect ration and update our car_id dict with the id of the object and the image, this helps us choose the biggest avilable photo.

## Part 2 training a classification model

I used resnet50 for this project which is a more than capbable model for this task but unfortuantley the available data I found was not sufficent, I wanted to use a recent dataset but the only dataset of quality on this subject was the stanford cars dataset which is about 5 years old.
I did however find a dataset that was recent on kaggle but the quality of the dataset was not optimal, there was not enough pictures considering how many our target classes are and also the dataset contained a lot of images which were interiors of the cars. This understandable considering the dataset was not a funded project but an opensourced personal project on kaggle (thank you: EIMANTAS KULBE), however if you intend on using this project I suggest you find a better dataset and train the model on it. the training is straight forward and it's done in the file name train.ipynb.
After about 9 epochs of training I found that the model began to overfit and the accuracy rate on the validation data didn't exceed 0.7. Given that the dataset contains images scraped from google search, the validation data looks very similar like the training data which both look very unsimilar to our exctracted photos from filmed videos, so this model performed horribly on my test video.

## Part 3 classification
File: yolo.py
Finally we loop pver our car_id dict and classify each image, save how many times an image with the same model and make occured and then saving the result in a csv file