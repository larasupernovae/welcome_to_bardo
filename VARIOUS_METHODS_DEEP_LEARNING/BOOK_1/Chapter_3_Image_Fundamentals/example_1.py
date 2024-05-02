# IMAGE FUNDAMENTALS

# here we load an image named example.png and display it to our screen

import cv2
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions

image = cv2.imread("lovely_sunset.png")             # Read the image
image_2 = cv2.resize(image, (504, 756))             # Resize image
print(image.shape)
cv2.imshow("Image", image_2)                        # Show the image
cv2.waitKey(0)                                      # Display the image infinitely until any keypress

# acesses pixel at x=100 and y=20
(b,g,r) = image[20,100]