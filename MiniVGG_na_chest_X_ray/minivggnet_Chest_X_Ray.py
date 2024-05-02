# Chest_X_RAY 5856, 32x32 black and white images
# 2 classes - PNEUMONIA 4273, NORMAL 1583
# miniVGG NO DATA AUGMENTATION

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

from utilities.nn.cnn import MiniVGGNet
# apply SGD to optmize the parameters of the network
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
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the chest_xray from disk then scales the raw pixel intensities
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# Load the dataset and scale the raw pixel intensities to the range [0, 1]
(data, labels) = sdl.load(imagePaths, verbose=732)  # bilo 500
data = data.astype('float') / 255.0

### Convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

### Account for skew in the labeled data
class_totals = labels.sum(axis=0)
class_weight = class_totals.max() / class_totals

# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, stratify = labels, random_state=42)

# "ONE HOT ENCODING"
# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
#train_y = to_categorical(train_y)
test_y = LabelBinarizer().fit_transform(test_y)
#test_y = to_categorical(test_y)

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
# SGD optimizer
# SGD learning rate 0.05
optimizer = SGD(lr=0.05, decay = 0.05/30)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), class_weight=class_weight, batch_size=24, epochs=30, verbose=1)

### SAVE THE NETWORK TO THE DISK - this method takes the weights and the state of the optimizer and serializes them
# to the disk in the HDF5 format, easy to save and load later.
print("[INFO]: Serializing....")
model.save(args["model"])

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=24)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 30), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 30), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 30), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 30), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on chest_xray minivggnet")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\MiniVGG_na_chest_X_ray
# $ python minivggnet_Chest_X_Ray.py --dataset chest_xray --model chest_x_ray_weights_1.hdf5