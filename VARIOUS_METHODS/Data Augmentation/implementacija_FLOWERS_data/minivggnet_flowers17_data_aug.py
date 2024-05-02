# Flowers-17 1360, 32x32 rgb images
# 17 classes - each class is 80 images
# miniVGG NO DATA AUGMENTATION

# VGGNet network uses entirely of 3x3 filters on CONV layers and they
# stack multiple CONV => ReLU layer sets (normally increasing the deeper
# we go before applying the POOL operation)

# "ONE HOT ENCODING" integer to vector labels
from sklearn.preprocessing import LabelBinarizer
# function used to help us train and test splits
from sklearn.model_selection import train_test_split
# for evaluating the performance of the classifier and print a table of results to our console
from sklearn.metrics import classification_report
# these 3 form a pipeline to process the images before passing them through the net
# accepts input image and converts it to a NumPy array that Keras can work with
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader

from utilities.nn.cnn import MiniVGGNet
# apply SGD to optmize the parameters of the network
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
# one switch --dataset, which is the path to the flowers-17 dataset directory
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we will be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# to extract the class label simply extract the 2nd to the last index
# flower17/bluebell/image_0232.jpg
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
# demonstrate the unique set of class labels from the image paths (17 classes)
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the Flowers-17 from disk then scales the raw pixel intensities
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# Load the dataset and scale the raw pixel intensities to the range [0, 1]
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# "ONE HOT ENCODING"
# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
#train_y = to_categorical(train_y)
test_y = LabelBinarizer().fit_transform(test_y)
#test_y = to_categorical(test_y)

# ImageDataGenerator
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
# randomly rotated +-30 degrees
# horizontally and vertically shifted by a factor of 0.2
# sheared by 0.2
# zoomed by uniformly sampling in the range [0.8, 1.2]
# randomly horizontally flipped

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
# SGD optimizer
# SGD learning rate 0.05
optimizer = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=32), validation_data=(test_x, test_y), steps_per_epoch=len(train_x) // 32, epochs=100, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Flowers-17 (with data augmentation")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\DEEP_LEARNING_CNN\VARIOUS METHODS\Data Augmentation\implementacija_FLOWERS_data
# $ python minivggnet_flowers17_data_aug.py --dataset 17flowers