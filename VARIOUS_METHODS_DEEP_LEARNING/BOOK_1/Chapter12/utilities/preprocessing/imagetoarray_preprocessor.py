# imports the image to array function from Keras
from keras.preprocessing.image import img_to_array
# accepts an input image and then converts it to NumPy array that Keras can work with

class ImageToArrayPreprocessor:
# define the constructor to our class - the constructor accepts an optional parameter named dataFormat
# the value is set to None which indicates that the setting inside keras.json should be used
    def __init__(self, data_format=None):
        # Store the image data format
        self.data_format = data_format

# accepts an image as input, calls the ordering the channels based on configuration value of dataFormat
# returns a new NumPy array with the cahnnels properly ordered
    def preprocess(self, image):
        # Apply the Keras utility function that correctly rearranges the dimensions of the image
        return img_to_array(image, data_format=self.data_format)
