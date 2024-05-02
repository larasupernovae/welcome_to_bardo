# import the necessary packages
import cv2

class SimplePreprocessor:
    # Method: Constructor define it (additional parametar for interpolation algorithm - which one is used)
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """
        :param width: Image width
        :param height: Image height
        :param interpolation: Interpolation algorithm
        """
        self.width = width
        self.height = height
        self.interpolation = interpolation

    # Method: Used to resize the image to a fixed size (ignoring the aspect ratio)
    # loading the image
    def preprocess(self, image):
        """
        :param image: Image
        :return: Re-sized image
        """
        # returns the resized image to the calling function
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
