# Identifying car models from a video using YOLO for car identification and tracking and ResNet50 for classification
## How the program works
This program takes a video of filmed cars and outputs a CSV file with car models and makes and how many times each model-make is identified.

## Part 1: Car identification and tracking
File: yolo.py

We use YOLOv8 for car identification and tracking. First, we create an empty car_id dictionary and an empty aspect_ratio_id dictionary. We then read the frame and extract the boxes, ids, names, and confidence scores. We zip them together and loop over each one. If the name is a car or a truck, we calculate the aspect ratio of the image extracted by the boxes from the frame. We then check if the aspect ratio is bigger than 1.15 and smaller than 2.5. This helps us to only select frames where a whole car is present in the image because sometimes YOLO will identify a car while it's not yet fully visible. We also check if the confidence score is higher than 0.7. This helps us to weed out any blurry extracted images because if the videos are filmed by hand, a lot of frames will have blurry images.

We then check the object id in our aspect_ratio_id dictionary and see what aspect ratio is currently saved for that id. We compare it against the current aspect ratio of the image we are inspecting. If the aspect ratio of the current image is bigger than the aspect ratio currently saved for that id in our aspect_ratio_id dictionary, we update our aspect_ratio_id with the new aspect ratio and update our car_id dictionary with the id of the object and the image. This helps us choose the biggest available photo.

## Part 2: Training a classification model
I used ResNet50 for this project, which is a more than capable model for this task. Unfortunately, the available data I found was not sufficient. I wanted to use a recent dataset, but the only dataset of quality on this subject was the Stanford Cars dataset, which is about 5 years old. I did, however, find a recent dataset on Kaggle, but the quality of the dataset was not optimal. There were not enough pictures considering how many our target classes are, and the dataset contained a lot of images that were interiors of the cars. This is understandable considering the dataset was not a funded project but an open-sourced personal project on Kaggle (thank you: EIMANTAS KULBE). However, if you intend on using this project, I suggest you find a better dataset and train the model on it. The training is straightforward and is done in the file named train.ipynb. After about 9 epochs of training, I found that the model began to overfit, and the accuracy rate on the validation data didn't exceed 0.7. Given that the dataset contains images scraped from Google search, the validation data looks very similar to the training data, which both look very dissimilar to our extracted photos from filmed videos. So, this model performed horribly on my test video.

## Part 3: Classification
File: yolo.py

Finally, we loop over our car_id dictionary and classify each image, save how many times an image with the same model and make occurred, and then save the result in a CSV file.






