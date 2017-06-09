import cv2 # OpenCV for perspective transform
import numpy as np
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt2
#import scipy.misc # For saving images as needed
#import glob  # For reading in a list of images from a folder

example_grid = '../calibration_images/example_grid1.jpg'
example_rock = '../calibration_images/example_rock1.jpg'
example_blue = '../misc/frame.png'
grid_img = cv2.imread(example_grid)
rock_img = cv2.imread(example_rock)
blue_img = cv2.imread(example_blue)

#flags = [i for i in dir(cv2) if i.startswith('COLOR_BGR')]
#print (flags)





# example works
# hsv = cv2.cvtColor(blue_img, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
# res = cv2.bitwise_and(blue_img,rock_img, mask= mask)

# define range of blue color in HSV


hsv = cv2.cvtColor(rock_img, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([0,100,100])
upper_yellow = np.array([176,255,255])

# Threshold the HSV image to get only yellow colors
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# Bitwise-AND mask and original image
result = cv2.bitwise_and(rock_img,hsv, mask= mask)


cv2.imshow('camera image of rock',rock_img)
cv2.imshow('mask',mask)
cv2.imshow('result',result)


cv2.waitKey(30000)
cv2.destroyAllWindows()


