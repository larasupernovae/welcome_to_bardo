import argparse
# for numerical processing
import numpy as np
import matplotlib.pyplot as plt

# when testing and impementing your own models from scratch
from sklearn.datasets import make_blobs
# for evaluating the performance of the classifier and print a table of results to our console
from sklearn.metrics import classification_report
# function used to help us train and test splits
from sklearn.model_selection import train_test_split


# Method: Used to compute the sigmoid activation value ON: >0.5  OFF: <0.5
def sigmoid_activation(x):
    """
    :param x: Feature matrix
    :return: Predictions matrix
    """
    return 1.0 / (1 + np.exp(-x))


# Method: Used to obtain a set of predictions
def predict(x, w):
    """
    :param x: Feature matrix
    :param w: Weights matrix
    :return: Predictions matrix
    """
    # Take the dot product between the features and weight matrices to get the prediction matrix
    predictions = sigmoid_activation(x.dot(w))

    # Apply a step function threshold for the binary outputs
    # class labels
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0] = 1 # ovo je uredu svakako gore imas uslov za manje od 0.5 najpre

    # return the predictions
    return predictions


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# number of epochs we will use
ap.add_argument('-e', '--epochs', required=False, type=float, default=100,
                help='# of epochs')
# learning rate - typically 0.1,0.01,0.001 but it is a hyperparamter - podesi ga za svoj problem sama
ap.add_argument('-a', '--alpha', required=False, type=float, default=0.01,
                help='learning rate')
args = vars(ap.parse_args())

# DATA FOR CLASSIFICATION

# Generate a 2-class 2D classification problem with 1,000 data points
# where each data point is a 2D feature vector
# labels for data points are 0 or 1 - train it to predict the class label for each point???
(x, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0]), 1)

# Insert a column of 1's as the last entry of the feature matrix (bias trick)
# now b is a trainable parameter within W
x = np.c_[x, np.ones((x.shape[0]))]

# Split the data into training (50%) and testing (50%)
(train_x, test_x, train_y, test_y) = train_test_split(x, y, test_size=0.5, random_state=42)

# Initialize the weight matrix and list of losses
print('[INFO]: Training....')
w = np.random.randn(x.shape[1], 1)
# initializes a list to keep track of our losses after each epoch
losses = []

# TRAINING AND GRADIENT DESCENT

# Loop for the number of epochs - so to see each of the training points a 100 times
for epoch in np.arange(0, args['epochs']):
    # Take the dot product between the features and weight matrices to get the predictions matrix
    predictions = sigmoid_activation(train_x.dot(w))

    # Compute the error between the predictions and true values d-y
    error = predictions - train_y
    # computes the least squared error - for binary classification problems
    # goal is to minimize our least squared error
    loss = np.sum(error ** 2)
    # memorize losses to later plot the function
    losses.append(loss)

    # Compute the gradient (dot product between the features and prediction errors)
    gradient = train_x.T.dot(error)

    # Update the weight matrix by 'nudging' it in the negative direction
    # we update in the negative direction of the gradient descent the W - we move to the bottom of the basin
    # CHAD THE ROBOT MOVES
    w += -args['alpha'] * gradient


    # Check to see if an update should be displayed
    # racuna ostatak % ako je on 0 nastavlja dalje jer je 100 sve sto se podeli sa 5 ako ima 0 pise
    # loss i baca posle na plot - znaci na svakih 5 baca mi update
    if epoch == 0 or (epoch+1) % 5 == 0:
        print('[INFO]: epoch={}, loss={:.4f}'.format(int(epoch + 1), loss))
# gradient descent is an iterative algorithm

# Evaluate the model
print('[INFO]: Evaluating....')
predictions = predict(test_x, w)
print(classification_report(test_y, predictions))

# Plot the classification (test) data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(test_x[:, 0], test_x[:, 1], marker='o', s = 30)
plt.show()

# Plot the loss over time
plt.style.use('ggplot')
plt.figure()
plt.title('Training Loss')
plt.plot(np.arange(0, args['epochs']), losses)
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()

# better to train it with more epoch, and smaller learning rate, beacuse it only has a W update every epoch
# START HERE
