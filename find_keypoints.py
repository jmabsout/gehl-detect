from collections import deque
import numpy as np
import imutils
import cv2
import datetime
from imutils.object_detection import non_max_suppression
import json


# TODO:
# think about the site in terms of 5'x5' grid cells (don't need to necessarily set one up but think about how data will be represented relative to that)
# first objective is to be able to count the number of people in the space at a given time...
# set up a counter or something, create a list of all the counts at a given interval, maybe average that? and have that
# be the count of people for a given period of time

vidPath = "./t3/GOPR2201.MP4"
video = cv2.VideoCapture(vidPath)

imagePath = "/Users/Mario/Documents/mit-github-projects/gehl/Gehl/t3/G0082212.JPG"

img = cv2.imread(imagePath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)



