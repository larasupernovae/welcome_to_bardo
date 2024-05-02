# import the necessary packages
import os
# to extract the names of subdirectories in image paths
import cv2
# openCV
import numpy as np
# for numerical processing

class SimpleDatasetLoader:
    # Method: Constructor
    # passing on the list of image processors that can be applied to a given input image
    def __init__(self, preprocessors=None):
        """
        :param preprocessors: List of image preprocessors
        """
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the processors are none instalise them as a empty list
        if self.preprocessors is None:
            self.preprocessors = []

    # Method: Used to load a list of images for pre-processing
    def load(self, image_paths, verbose=-1):
        """
        :param image_paths: List of image paths
        :param verbose: Parameter for printing information to console
        :return: Tuple of data and labels
        """
        # instalize the list of features and labels aka images themselves
        data, labels = [], []

        # single PARAMETER - image_path required for the load method
        # loop over the input images
        for (i, image_path) in enumerate(image_paths):

            # load the image and extract the class label assuming our path has
            # following format /path/to/dataset/{class}/{image}.jpg
            # exmp: /animals/panda/image.jpg
            image = cv2.imread(image_path)
            # extracting the class label based on the file path
            label = image_path.split(os.path.sep)[-2]

            # check to see if the preprocessors are not None
            if self.preprocessors is not None:

                # loop over the preprocessor and apply each to the image
                # apply them to the image
                for p in self.preprocessors:
                    # this action allows us to form a chain of preprocessors
                    # that can be applied to every image in a dataset
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # once the image has been preprocessed we update the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every 'verbose' image
            # 'verbose' - print updates to a console
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print('[INFO]: Processed {}/{}'.format(i+1, len(image_paths)))

        # return the tuple of the data and labels
        return (np.array(data), np.array(labels))
