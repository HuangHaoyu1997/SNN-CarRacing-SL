# SNN-CarRacing-SL
train a SNN model using supervised learning to handle CarRacing problem

We use first a CNN model to train the dataset collected by human playing in CarRacing Game. Then we use a Spike-CNN model to train the dataset and compare performance on two models.

The model trained by dataset collected by human-playing CarRacing Game Simulator is tested on the env of `car_racing.py` and the average score is around 670.

Our dataset is ~80000 images with corresponding action labels. We split the total dataset into training set and testing set on the ratio of 4:1.
