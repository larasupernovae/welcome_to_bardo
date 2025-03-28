from keras.models import Sequential
# implemetation of the convolutional layer
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
# applying the activation function to an input
from keras.layers.core import Activation
# takes the multi-dimensional volume and "flatens" it into 1D array prior to feeding the inputs into the Dense layer
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
     # width of the input image
     # height of the input image
     # depth - number of cannels in the input image
     # classes - number of classes our network needs to learn to predict

       # Initialize the model - building block of sequential networks sequentially stack one layer on top of the other
        model = Sequential()
        input_shape = (height, width, depth)

        # If we are using 'channels-first', update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # First set of CONV => ReLU => POOL layers
        # convolutional layer will learn 20 filters size FxF = 5x5
        # 2x2 pooling layer with stride 2x2 --> decreasing the input volume size
        # by 75%
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second set of CONV => RELU => POOL layers
        # convolutional layer will learn 50 filters size FxF = 5x5
        # 2x2 pooling layer with stride 2x2 --> decreasing the input volume size
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # First (and only) set of FC => RELU layers
        # 500 fully connected layer with 500 nodes
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model
