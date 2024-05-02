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


# Method: Used to compute the sigmoid activation value
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
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0] = 1

    return predictions


# Method: Used to get the next 'mini-batch' of data
# mini-batch are typically 32,64,128,256 => why? more stable convrgance
def next_batch(x, y, batch_size):
    """
    :param x: Feature matrix - of our pixeleted images
    :param y: Class matrix - class lables associated with each of the training points
    :param batch_size: Batch size - size of the mini batch that will be returned
    :return: Subsets of x and y as mini-batches
    """
    # Loop over our data in 'mini-batches' yielding a tuple of the current batched data
    for i in np.arange(0, x.shape[0], batch_size):
        yield (x[i:i + batch_size], y[i:i + batch_size])


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', required=False, type=float, default=100,
                help='# of epochs')
ap.add_argument('-a', '--alpha', required=False, type=float, default=0.01,
                help='learning rate')
# make the default 32 data points per mini-batch
ap.add_argument('-b', '--batch_size', required=False, type=int, default=32,
                help='Size of SGD mini-batches')
args = vars(ap.parse_args())

# Generate a 2-class 2D classification problem with 1,000 data points - adding a bias column, performing training and testing
(x, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0]), 1)

# Insert a column of 1's as the last entry of the feature matrix (bias trick)
x = np.c_[x, np.ones((x.shape[0]))]

# Split the data into training (50%) and testing (50%)
(train_x, test_x, train_y, test_y) = train_test_split(x, y, test_size=0.5, random_state=42)

# Initialize the weight matrix and list of losses
print('[INFO]: Training....')
w = np.random.randn(x.shape[1], 1)
losses = []

# Loop for the number of epochs - LOOP OVER THE DESIRED NUMBER OF EPOCHS, sampling mini-batch along the way
for epoch in np.arange(0, args['epochs']):
    # Initialize the total loss for each epoch
    epoch_loss = []

    # we loop over our traing data in the batch (SMALL PART OF THE TRAINING SET!)
    for (batch_x, batch_y) in next_batch(x, y, args['batch_size']):
        # Take the dot product between the current batch features and the weight matrix to get the predictions matrix
        predictions = sigmoid_activation(batch_x.dot(w))

        # Compute the error between the predictions and true values
        error = predictions - batch_y
        epoch_loss.append(np.sum(error ** 2))

        # Compute the gradient (dot product between the current batch features and prediction errors)
        gradient = batch_x.T.dot(error)

        # Update the weight matrix by 'nudging' it in the negative direction
        w += -args['alpha'] * gradient
# !!! NOTICE: see how the weight update takes place inside the batch loop - this implies
# there are multiple weight updates per epoch


    # Update the loss history by averaging the loss over each batch
    loss = np.average(epoch_loss)
    losses.append(loss)

    # Check to see if an update should be displayed
    if epoch == 0 or (epoch+1) % 5 == 0:
        print('[INFO]: epoch={}, loss={:.4f}'.format(int(epoch + 1), loss))


# Evaluate the model
print('[INFO]: Evaluating....')
predictions = predict(test_x, w)
print(classification_report(test_y, predictions))

# Plot the classification (test) data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(test_x[:, 0], test_x[:, 1], marker='o', s=30)
plt.show()

# Plot the loss over time
plt.style.use('ggplot')
plt.figure()
plt.title('Training Loss')
plt.plot(np.arange(0, args['epochs']), losses)
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()

# START HERE