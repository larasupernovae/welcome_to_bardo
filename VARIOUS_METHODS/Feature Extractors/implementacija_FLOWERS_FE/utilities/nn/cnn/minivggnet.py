from keras.models import Sequential
# implements batch normalization
from keras.layers.normalization import BatchNormalization
# implemetation of the convolutional layer
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
# applying the activation function to an input
from keras.layers.core import Activation
# takes the multi-dimensional volume and "flatens" it into 1D array prior to feeding the inputs into the Dense layer
from keras.layers.core import Flatten
# implements dropout
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Initialize the model, input shape and the channel dimension
        model = Sequential()
        input_shape = (height, width, depth)
        # imports a variable - the index of the channel dimension - batch normalization operates
        # over channels so in order to apply BN, we need to know which axis to normalize over
        # if it is -1 it implies that the index of the channel is the last in the input shape
        # if it is 1 the channel is first
        channel_dim = -1

        # If we are using 'channels_first', update the input shape and channels dimension
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
            channel_dim = 1

        # First CONV => RELU => CONV => RELU => POOL layer set
        # 32 filters, 3x3 filter size
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        # fed into a batchnormalization layer to zero-center the activations
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        # instead of the pool 2 sets for deepr CNN
        # dropout with the probability of p = 0.25, it implies that the node
        # from the POOL layer will randomly disconnect from the next layer with the
        # probability of 25% during training
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # DROPOUT USED TO REDUCE THE EFFECTS OF OVERFITING
        model.add(Dropout(0.25))

        # Second CONV => RELU => CONV => RELU => POOL layer set
        # WE ARE LEARNING 2 SETS OF 64 FILTERS (SIZE 3x3)
        # IT IS COMMON TO INCREASE THE NUMBER OF FILTERS AS THE SPATIAL INPUT
        # SIZE DECREASES DEEPER IN THE NETWORK
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # First (and only) set of FC => RELU layers
        # dropout typically p = 0.5 applied between FC layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # Return the constructed network architecture
        return model
