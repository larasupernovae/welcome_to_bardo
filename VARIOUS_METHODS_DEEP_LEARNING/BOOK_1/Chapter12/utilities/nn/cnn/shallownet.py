from keras.models import Sequential
# implemetation of the convolutional layer
from keras.layers.convolutional import Conv2D
# applying the activation function to an input
from keras.layers.core import Activation
# takes the multi-dimensional volume and "flatens" it into 1D array prior to feeding the inputs into the Dense layer
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

# build method - for every CNN
# this function will accept a number of parameters construct the network architecture and then return it to the calling function

class ShallowNet:
    @staticmethod
    # width - number of columns
    # height - number of rows
    # depth - number of cannels in the input image
    # classes - number of classes our network needs to learn to predict
    def build(width, height, depth, classes):
        # Initialize the model along with the input shape to be 'channels_last'
        model = Sequential()
        input_shape = (height, width, depth)

        # Update the image shape if 'channels_first' is being used
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        # this layer will have 32 filters each of 3x3 FxF - applying the same padding to ensure the size of the output
        # of the convolutional operation matches the inuput
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        # ReLU activation function
        model.add(Activation('relu'))

        # flatten the multi dimensional representation into a 1D list
        model.add(Flatten())
        # uses the same number of nodes as our output class label
        model.add(Dense(classes))
        # applies softmax which will give us class label probabilities for each class
        model.add(Activation('softmax'))

        # Return the network architecture
        return model
