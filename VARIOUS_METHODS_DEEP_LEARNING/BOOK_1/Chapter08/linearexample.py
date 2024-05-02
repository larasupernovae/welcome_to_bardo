# to load our example image from the disc
import cv2
# numerical processing
import numpy as np

# Initialize class labels and set the seed of our pseudo-random number generator
labels = ['dog', 'cat', 'panda']
# '1' is chosen as the seed because it gives the 'correct classification'
np.random.seed(1)

# Randomly initialize the weight and bias vectors between 0 and 1
# uniformed distribution 32x32x3 pixels 3072 and 3 classes
w = np.random.randn(3, 3072)
# same, for 3 classes
b = np.random.randn(3)

# Load image via cv2.imread
# resize it (ignoring the aspect ratio) and flatten it
original = cv2.imread('beagle.png')
# flatten into 3072 vector
image = cv2.resize(original, (32, 32)).flatten()

# Compute the output scores
scores = w.dot(image) + b

# Loop over the scores and labels to display them
for label, score in zip(labels, scores):
    print('[INFO]: {}: {:.2f}'.format(label, score))

# Draw the label with the highest score on the image as our prediction
cv2.putText(original, 'Label: {}'.format(labels[np.argmax(scores)]), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display our input image
cv2.imshow("Image", original)
cv2.waitKey(0)

# COMMAND PROMPT
# $ cd \Users\jopas\OneDrive\Desktop\6 Deep Learning for Computer Vision Adrian Rosebrock\Chapter08
# $ python linearexample.py