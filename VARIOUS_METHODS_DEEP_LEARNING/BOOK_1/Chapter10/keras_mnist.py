# MNIST dataset = 70000 data points
# 7000 examples per digit - each data point is represented by a 784 vector
# corresponding to a flattened 28x28 images = 784

# "ONE HOT ENCODING" integer to vector labels
from sklearn.preprocessing import LabelBinarizer
# training and test splits from dataset
from sklearn.model_selection import train_test_split
# nice formated display of total accuracy of our model for each digit beakdown
from sklearn.metrics import classification_report

# SIMPLE FEEDFORWARD NN with Keras
# network is feedforward and layers will be added to class sequentially
from keras.models import Sequential
# implementaition of fully connected layers
from keras.layers.core import Dense
# apply SGD to optmize the parameters of the network
from keras.optimizers import SGD
# to gain accesss to the full MINST dataset
from sklearn import datasets
from sklearn.datasets.mldata import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
import argparse


# FOR WIFI CONNECTION! to download the dataset
#import requests

#s = requests.Session()
#s.proxies = {"https": "https://87.116.178.8"}

#r = s.get("http://www.google.com")
#print(r.text)

# Construct the argument parser and parse the arguments
# single swith --output, it is the path to where our plotting and loss accuaracy will be saved on the dick
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
                help='Path to the output loss/accuracy plot')
args = vars(ap.parse_args())

# Get the MNIST dataset
print('[INFO]: Loading the MNIST (full) dataset....')
dataset = datasets.fetch_mldata('MNIST Original')
#mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
#dataset = mnist

# DATA NORMALIZATION - scale the raw pixel intensities to the range [0, 1], then construct the training and testing splits
data = dataset.data.astype('float') / 255.0
(train_x, test_x, train_y, test_y) = train_test_split(data, dataset.target, test_size=0.25)
# train 75% test 25%

# Convert the labels from integers to vectors "one hot encoding"
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

# Define the 784-256-128-10 achitecture using keras
# layers will be stacked on top of each other
model = Sequential()
# first fully conected layer 28x28 = 784 , 256 weights
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
# 10 weights (zero to 9) switch to softmax
model.add(Dense(10, activation='softmax'))

# Train the model using SGD
print('[INFO]: Training....')
# initialize SGD optimizer with learning rate 0.01
sgd = SGD(0.01)
# cross-entropy function and loss
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=128)
# 100 epochs using a BATCH SIZE of 128 data points at a time

# Test the network
# will return the class label probabilities or every data point in testX
# (X,10) there are 17500 total data points in the testing set and 10 possible classes
print('[INFO]: Testing....')
predictions = model.predict(test_x, batch_size=128)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

plt.style.use('ggplot')
#plt.figure()
#plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
#plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
#plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
#plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
#plt.title('Training Loss & Accuracy')
#plt.xlabel('Epoch #')
#plt.ylabel('Loss/Accuracy')
#plt.legend()
#plt.savefig(args['output'])
#plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter10
# $ python keras_mnist.py --output output/keras_mnist.mat