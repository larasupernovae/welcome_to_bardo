# import the necessary packages
from imutils import paths
# k-NN algorithm provided by cikit-learn library
from sklearn.neighbors import KNeighborsClassifier
# to convert labels from string to integers, beacuse each class is represented with an integer
from sklearn.preprocessing import LabelEncoder
# function used to help us train and test splits
from sklearn.model_selection import train_test_split
# for evaluating the performance of the classifier and print a table of results to our console
from sklearn.metrics import classification_report
import argparse

# implementations of the previous 2 .py scripts
from utilities.preprocessor import SimplePreprocessor
from utilities.dataset_loader import SimpleDatasetLoader

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# the path to where our imput image dataset resides on disk
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
# the number of neighbors k to apply when using the K-NN algorithm
ap.add_argument('-k', '--neighbors', required=False, type=int, default=1,
                help='# of nearest neighbors for classification')

# for computing the distance between an input data point and the training set, value of -1 will use all available cores on the processor
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1,
                help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

# STEP 1: grab the file pathes of the images in our data
# Get list of image paths
image_paths = list(paths.list_images(args['dataset']))

# Initialize SimplePreprocessor and SimpleDatasetLoader and load dataset from disk and labels
# and reshape the data matrix
print('[INFO]: Images loading....')
sp = SimplePreprocessor(32, 32)   # rekli smo vec da cemo da ih smanjimo na 32x32
# this implyes that sp will be applied to every image in the dataset
sdl = SimpleDatasetLoader(preprocessors=[sp])
# loads the actual image dataset from the disk - it returns each image resized to 32x32
# pixels along with the labels
(data, labels) = sdl.load(image_paths, verbose=500)

# the data NumPy array is
# Reshaped from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
# this means that there are 3000 images 32x32 size with 3 channels
# "flatten" images to from a 3D to a single list (3000,3072) , 3072 = 32x32x3
data = data.reshape((data.shape[0], 3072))

# demostrating how much memory it takes to store these 3000 images, bytes, and converts to megabytes
# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(float(data.nbytes/(1024*1000))))

# STEP 2: train and test splits
# Encode labels as integers
# converts the labels from string to integer, cats = 0, dogs = 1, pandas = 2
le = LabelEncoder()
labels = le.fit_transform(labels)

# X - refers to data points, Y - class labels
# Split data into training (75%) and testing (25%) data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# STEP 3,4: create K-NN classifier and evaluate it
# Train and evaluate the k-NN classifier on the raw pixel intensities
print('[INFO]: Classification starting....')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])

# no actuall learning, only storing so it can compute the distance between the input data and trainX
model.fit(train_x, train_y)
# testY class, predicted class labels, name of class labels
print(classification_report(test_y, model.predict(test_x),
                            target_names=le.classes_))

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter07
# $ python knn.py --dataset animals