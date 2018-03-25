# Standard imports
import cv2
import numpy as np
import glob

out_file = './preprocessing/test/'
kernel = np.ones((13,13), np.uint8)
lower_blue = np.array([110, 245, 63])
upper_blue = np.array([130, 265, 143])


for filename in glob.glob('./preprocessing/0001-bird-eye-view/*.png'):
    img = cv2.imread(filename, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    img_dilation = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite('{}{}'.format(out_file, filename[-7:]), img_dilation)
    # cv2.imshow('frame', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    # cv2.imshow('mask_good', img_good)

cv2.waitKey(0)