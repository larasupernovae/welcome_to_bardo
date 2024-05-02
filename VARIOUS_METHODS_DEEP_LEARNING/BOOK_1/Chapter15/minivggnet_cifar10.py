# CIFAR 10 - 60000, 32x32 rgb images
# 10 classes - each class is 6000 images
# 50000 training images, 10000 testing images

# VGGNet network uses entirely of 3x3 filters on CONV layers and they
# stack multiple CONV => ReLU layer sets (normally increasing the deeper
# we go before applying the POOL operation)

# reduces overfitting and increase our classification accuracy

# Set the matplotlib backend so figures can be saved in the background
# create a non-interactive that will simply be saved to the disk
import matplotlib
matplotlib.use("Agg")

# "ONE HOT ENCODING" integer to vector labels
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from utilities.nn.cnn import MiniVGGNet
# apply SGD to optmize the parameters of the network
from keras.optimizers import SGD
# this is a helper function to automatically load this dataset from disk
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Construct the argument parser and parse the arguments
# path --output to our output training and loss plot
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# Load the cifar10 dataset, pre split into training and testing data,
# then scale it into the range [0, 1]
print("[INFO]: Loading CIFAR-10 data....")
((train_x, train_y), (test_x, test_y)) = cifar10.load_data()
train_x = train_x.astype("float") / 255.0
test_x = test_x.astype("float") / 255.0

# Convert the labels from integers to vectors "ONE HOT ENCODING"
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# Initialize the label names for the CIFAR-10 dataset
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
# instead of SGD Nestrov accelerated gradient added to the SGD (over time changes the learing rate slows down)
# SGD learning rate 0.01
# momentum gama = 0.9
# DECAY is to the divide the initial learning rate with the number of epochs
optimizer = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
# same as before
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=64, epochs=40, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=64)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10 (with Batch Normalization")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter15
# $ python minivggnet_cifar10.py --output output/cifar-10-python.tar