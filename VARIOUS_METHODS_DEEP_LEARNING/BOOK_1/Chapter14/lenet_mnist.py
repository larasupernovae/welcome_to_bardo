# MNIST dataset 28x28 input image 784 vectors
# 70000 data points
# 7000 examples per digit - each data point is represented by a 784 vector
# corresponding to a flattened 28x28 images = 784

# INPUT (28x28x1) => CONV (28x28x20) 5x5,K=20 filters => ReLU (28x28x20) =>
# POOL (14x14x20) 2x2 => CONV (28x28x50) 5x5,K=50 => ReLU (14x14x50) =>
# POOL (7x7x50) 2x2 => FC (500 hidden nodes) => ReLU (500) =>
# FC (10) for each class 0 to 9 => Softmax (10)

from utilities.nn.cnn import LeNet
# apply SGD to optmize the parameters of the network
from keras.optimizers import SGD
# "ONE HOT ENCODING" integer to vector labels
from sklearn.preprocessing import LabelBinarizer
# function used to help us train and test splits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
# for the Keras modules you write to be compatible with both Theano (th) and TensorFlow (tf),
# you have to write them via the abstract Keras backend API
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# Grab the MNIST dataset
print('[INFO]: Accessing MNIST....')
dataset = datasets.fetch_mldata('MNIST Original')
data = dataset.data

# SHAPE OUR MATRIX
# 'channels_first' ordering
if K.image_data_format() == "channels_first":
    # Reshape the design matrix such that the matrix is: num_samples x depth x rows x columns
    data = data.reshape(data.shape[0], 1, 28, 28)
# 'channels_last' ordering
else:
    # Reshape the design matrix such that the matrix is: num_samples x rows x columns x depth
    data = data.reshape(data.shape[0], 28, 28, 1)

# TRAIN AND TEST 75% and 25%
# Scale the input data to the range [0, 1] and perform a train/test split
(train_x, test_x, train_y, test_y) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)

# Convert the labels from integers to vectors "One HOT encoding"
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

# train LeNet on MNIST
# Initialize the optimizer and model
print("[INFO]: Compiling model....")
# SGD optimizer with learning rae 0.01
optimizer = SGD(lr=0.01)
# 28x28 pixesls and depth of 1
model = LeNet.build(width=28, height=28, depth=1, classes=10)
# compile the model using cross-entropy loss as our loss function
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
# trains LeNet on MNIST 20 epochs using mini batch size of 128
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=128, epochs=20, verbose=1)

# Evaluate the network
print("[INFO]: Evaluating....")
# SO WHEN THE model.predict IS CALLED FOR EACH SAMPLE IN testX BATCH SIZES OF 128 ARE CONSTRUCTED
# AND THEN PASSED THROUGH THE NETWORK FOR CLASSIFICATION
# after all testing data points are classified the prediction variable is returned
# THE PREDICTION VARIABLE IS A NumPy ARRAY WITH THE SHAPE OF (len(testX),10), WE NOW
# HAVE 10 PROBABILITIES ASSOCIATED WITH EACH CLASS LABEL FOR EVERY DATA POINT IN testX
predictions = model.predict(test_x, batch_size=128)
# taking axis=1 it finds the index of the label with the largest probability, and we compare the
# predicted and ground truth labels later on
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter14
# $ python lenet_mnist.py --output output/keras_mnist.mat