# "ONE HOT ENCODING" integer to vector labels
from sklearn.preprocessing import LabelBinarizer
# function used to help us train and test splits
from sklearn.model_selection import train_test_split
# for evaluating the performance of the classifier and print a table of results to our console
from sklearn.metrics import classification_report
# these 3 form a pipeline to process the images before passing them through the net

# accepts input image and converts it to a NumPy array that Keras can work with
from utilities.preprocessing import ImageToArrayPreprocessor
# implementations of the previous 2 .py scripts
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader

# INPUT => CONV => RELU => FC
from utilities.nn.cnn import ShallowNet
# apply SGD to optmize the parameters of the network
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
# for numerical processing
import numpy as np
import argparse

# Construct the argument parser and parse the arguments
# one switch --dataset, which is the path to the animal set directory
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")

# second switch ADDED!!!
# switch --model the path to where we save network after the training is complete
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

# Grab the list of images
print("[INFO]: Loading images....")
image_paths = list(paths.list_images(args["dataset"]))

# Initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=100, verbose=1)

# SAVE THE NETWORK TO THE DISK - this method takes the weights and the state of the optimizer and serializes them
# to the disk in the HDF5 format, easy to save and load later.
print("[INFO]: Serializing....")
model.save(args["model"])

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

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
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter13
# $ python shallownet_train.py --dataset animals --model shallownet_weights.hdf5