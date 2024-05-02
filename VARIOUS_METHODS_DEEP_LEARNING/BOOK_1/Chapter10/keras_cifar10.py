# CIFAR 10 - 60000, 32x32 rgb images
# 10 classes - each class is 6000 images
# 50000 training images, 10000 testing images

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
# this is a helper function to automatically load this dataset from disk
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# Load the training and testing data, scale it into the range [0, 1], then reshape the design matrix
print("[INFO]: Loading CIFAR-10 dataset...")
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()

# convert the data type for unsugned 8-bit integers to floating point followed by the scaling the data
# to the range [0,1]
train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0
# for reshaping and training
# 32x32x3 = 3072
# flatten each image use reshape from NumPy to (50000,3072)
train_x = train_x.reshape((train_x.shape[0], 3072))
# flatten each image to (10000,3072)
test_x = test_x.reshape((test_x.shape[0], 3072))

# Convert the labels from integers to vectors "one hot encoding"
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Train the model using SGD
print("[INFO]: Training....")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=32)

# Test the network - epochs 100 batch size 32
print("[INFO]: Testing...")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter10
# $ python keras_cifar10.py --output output/cifar-10-python.tar