# covid19_CT 672, 32x32 color images
# 2 classes - CT_COVID 349, CT_NonCOVID 323
# miniVGG DATA AUGMENTATION

# VGGNet network uses entirely of 3x3 filters on CONV layers and they
# stack multiple CONV => ReLU layer sets (normally increasing the deeper
# we go before applying the POOL operation)

# "ONE HOT ENCODING" integer to vector labels
from sklearn.preprocessing import LabelBinarizer
# from keras.utils import to_categorical
# function used to help us train and test splits
from sklearn.model_selection import train_test_split
# for evaluating the performance of the classifier and print a table of results to our console
from sklearn.metrics import classification_report
# these 3 form a pipeline to process the images before passing them through the net
# accepts input image and converts it to a NumPy array that Keras can work with
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader

###
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from utilities.nn.cnn import LeNet
from keras.optimizers import Adam
# apply SGD to optmize the parameters of the network
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
# for numerical processing
import numpy as np
import argparse
import os

# Construct the argument parser and parse the arguments
# one switch --dataset, which is the path to the chest_xray dataset directory
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")

### second switch ADDED!!!
# switch --model the path to where we save network after the training is complete
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

# grab the list of images that we will be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# to extract the class label simply extract the 2nd to the last index
# chest_xray/NORMAL/image_010001.jpg
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
# demonstrate the unique set of class labels from the image paths (2 classes)
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the chest_xray from disk then scales the raw pixel intensities
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# Load the dataset and scale the raw pixel intensities to the range [0, 1]
(data, labels) = sdl.load(imagePaths, verbose=84)
data = data.astype('float') / 255.0

# Convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# Account for skew in the labeled data
class_totals = labels.sum(axis=0)
class_weight = class_totals.max() / class_totals

# Partition the data into training data (80%) and testing data (20%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

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

# Initialize the model
print("[INFO]: Compiling model....")
model = LeNet.build(width=32, height=32, depth=3, classes=len(classNames))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit_generator(aug.flow(train_x, train_y, batch_size=24), validation_data=(test_x, test_y), class_weight=class_weight, steps_per_epoch=len(train_x) // 24, epochs=30, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=24)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Save the model to disk
print("[INFO]: Serializing network....")
model.save(args["model"])

# Plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 30), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 30), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 30), H.history["acc"], label="acc")
plt.plot(np.arange(0, 30), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Documents\MEGA\DEEP_LEARNING_CNN\deep_CNN_covid19_dataset\covid19_miniVGG_LeNet
# $ python train_model_lenet.py --dataset covid19_CT_images --model lenet.hdf5