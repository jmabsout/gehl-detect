import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import os
import glob

imagePath = "/Users/Mario/Documents/mit-github-projects/gehl/Gehl/t3/G0082212.JPG"
imageDir = "/Users/Mario/Documents/mit-github-projects/gehl/Gehl/t3/"


images = []

os.chdir(imageDir)
for file in glob.glob("*.JPG"):
    images.append(file)
# print images
KNOWN_DISTANCE = 96.0
KNOWN_WIDTH = 5.0

# should be extended to read video
# should also be extended to read profile faces too (and/or a trained face finder) to better be able to find a person's distance based on different head dimensions

img8 = cv2.imread(imagePath)

# face_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades_cuda/haarcascade_profileface.xml")
face_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml")

num_matches = []
def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print type(face)https://drive.google.com/file/d/0B7cA2u-PMwdHdUt5S1U4bzdNQTA/view?usp=sharing
    if type(face) is tuple:
        rects = [[0,0,1,1],"noface"]
        print "no face"
        return rects
    else:
        for (x, y, w, h) in face:
            i = 0
            rect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])
            print "good face rects =  ",rects
            # rs = rects[0]
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            num_matches.append(i)
            i +=1
            return rects


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth



marker = find_marker(img8)
faceWidth = (marker[0][2]-marker[0][0])
print "t8 feet image marker: ", marker
print "face width in pixels: ", faceWidth
focalLength = (faceWidth * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for each in images:
    # print images
    image = cv2.imread(each)
    # print image
    # image = imutils.resize(image, width=min(1000, image.shape[1]))
    # print each
    marker = find_marker(image)
    # print marker
    # print marker[0]
    # print marker[0][0]
    faceWidth = (marker[0][2] - marker[0][0])
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, faceWidth)
    feet = inches/12
    print "distance: ", feet, " feet."

    cv2.putText(image, "%.8fft" % (inches / 12),
                (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)
    image = imutils.resize(image, width=min(1000, image.shape[1]))
    cv2.imshow("image", image)
    cv2.waitKey(1)
    print len(num_matches)


