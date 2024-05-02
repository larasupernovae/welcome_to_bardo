# LOAD A PRE-TRAINED MODEL AND USE IT ON IMAGES TO
# PUT THEM IN THEIR RESPECTIVE CLASS - classify individual images

# import the classes, standard pipeline
# -- resizing the image to a fixed size
# -- converting it to a Keras compatible array
# -- using preprocessors to load an entire image dataset into memory
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader

# function used to load our trained model from disk - it acceps the path to our trained network HDF5
# decoding the weights and optimizer, and setting the weights inside our architecture so we can
# continue training or use the network to classify new images
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
# OpenCV bindings - to display our images later to the screen
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# Initialize the class labels
classLabels = ["CT_COVID","CT_NonCOVID"]

# Grab a random sample of images from the dataset - CODE BLOCK RANDOM SAMPLING 10 IMAGE PATHS for dataset
print("[INFO]: Sampling images....")
image_paths = np.array(list(paths.list_images(args["dataset"])))
indexes = np.random.randint(0, len(image_paths), size=(30,))
image_paths = image_paths[indexes]

# Initialize the image preprocessors - all 10 images must be processed
sp = SimplePreprocessor(32, 32)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths)
data = data.astype("float") / 255.0

# !!! ALWAYS - take care to ensure that your testing images are preprocessed the same way as your training images

# Load the pre-trained network
print("[INFO]: Loading pre-trained network....")
model = load_model(args["model"])

# Make predictions on the images
print("[INFO] Predicting...")
# axis = 1 so it will only show you the highest probability for each image
predictions = model.predict(data, batch_size=32).argmax(axis=1)

# VISUALISE THE RESULTS
# loop over the sample images
for (i, image_path) in enumerate(image_paths):
    # Load the example image, draw the prediction, and display it
    image = cv2.imread(image_path)
    # draw the class label for each image
    cv2.putText(image, "Label: {}".format(classLabels[predictions[i]]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # display to screen
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Documents\MEGA\DEEP_LEARNING_CNN\deep_CNN_covid19_dataset\covid19_miniVGG_LeNet
# $ python covid19_load.py --dataset covid19_CT_images --model covid19_CT_100_weights.hdf5