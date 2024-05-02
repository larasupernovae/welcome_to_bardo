# MNIST - Sample not whole
# MNIST is bulit into scikit-learn library 1797 example digits
# 8x8 grayscale images (originals are 28x28) 8x8=64 vector

from utilities.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Load the MNIST dataset and apply min/max scaling to scale the pixel intensity values to the range [0, 1] (each
# image is represented as an 8x8 = 64-dim feature vector
digits = datasets.load_digits()
data = digits.data.astype('float')

# min-max normalizing by scaling each digit into the range [0,1]
data = (data - data.min()) / (data.max() - data.min())
print('[INFO]: Samples={}, Dimension={}'.format(data.shape[0], data.shape[1]))

# Construct the training and testing splits - evaluation 25% and testing 75%
(train_x, test_x, train_y, test_y) = train_test_split(data, digits.target, test_size=0.25)

# Convert the labels from integers to vectors - ONE HOT ENCODING
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

print(train_x.shape)
print(train_y.shape)

# Train the network
print('[INFO]: Training....')
# 64 - 32 - 16 - 10 = neural network architecture the output layer 10 nodes (zero to 9)
nn = NeuralNetwork([train_x.shape[1], 32, 16, 10])
print('[INFO]: {}'.format(nn))
nn.fit(train_x, train_y, epochs=1000)

# Test the network
print('[INFO]: Testing....')
# prediction for every data point - prediction array shape (450,10)
# 450 - testing set, and 10 (zero to 9)
predictions = nn.predict(test_x)

# ARGAX function - largest probability for each data point
# this function will return the index of the label with the highest predicted probability
predictions = predictions.argmax(axis=1)
# a nice classification report display
print(classification_report(test_y.argmax(axis=1), predictions))

# START HERE!
