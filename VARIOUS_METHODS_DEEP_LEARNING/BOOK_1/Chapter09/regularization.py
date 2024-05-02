import argparse
import numpy as np
from imutils import paths
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

# Get list of image paths
image_paths = list(paths.list_images(args['dataset']))

# Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
print('[INFO]: Images loading....')
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)

# Reshape from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
data = data.reshape((data.shape[0], 3072))

# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes / 1024*1000.0)))

# Encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split data into training (75%) and testing (25%) data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=5)

# LOOPS OVER REGULARIZERS

# cross-entropy loss with regulazation penalty of r and lambda = 0.0001
# learning rate 0.01 and 10 epochs
for reg_method in (None, 'l1', 'l2'):
    # Train a SGD classifier using a softmax loss function the specified regularization function
    # for 10 epochs
    print('[INFO]: Training model with {} penalty'.format(str(reg_method).upper()))

# BITNO!!! obrati paznju na max_iter na osnovu obima slika tj broja slika

    # INITIALIZING AND TRAING THE SGD Classifer
    model = SGDClassifier(loss='log', penalty=reg_method, max_iter=50,
                          learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(train_x, train_y)

    # Evaluate the classifier
    accuracy = model.score(test_x, test_y)
    print('[INFO]: {} penalty accuracy: {:.2f}%'.format(str(reg_method).upper(), accuracy*100))

# IF TUNED RIGHT IT REDUCES OVERFITTING AND BOOSTS THE TESTING ACCURACY
# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter09
# $ python regularization.py --dataset animals