import numpy as np
import imutils
import cv2
import datetime
from imutils.object_detection import non_max_suppression
import os
import glob
import gehl_detect as gd

cwd = os.getcwd()
print cwd

imageDir = '/Users/marioag/Dropbox (Personal)/documents/mit-github-projects/gehl/Gehl/t3/'


os.chdir(imageDir)
cwd = os.getcwd()
print cwd

images = []
images = gd.get_jpgs(imageDir)

# really silly to have to put full paths but ocv throws an error with relative paths, 
# or with os.path.abspath() concatenation...

# fface_cascade = cv2.CascadeClassifier('/Users/marioag/Documents/GitHub/gehl-detect/ocv2/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
fface_cascade = cv2.CascadeClassifier('/Users/marioag/Documents/GitHub/gehl-detect/ocv2/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
pface_cascade = cv2.CascadeClassifier('/Users/marioag/Documents/GitHub/gehl-detect/ocv2/opencv/data/haarcascades/haarcascade_profileface.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# print "starting loop over {} images...".format(len(images))

j = 0
cap = cv2.VideoCapture(0) # reads from webcam instead of directory of files
while(True):
    ret, image = cap.read()
# for each in images:
#     image = cv2.imread(each)
    # the image size is one of the most important parameters to tweak; the bigger it is,
    # the greater the chance that objects will be detected; however, it will result in 
    # slower processing time.
    image = imutils.resize(image, width=min(1500, image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fface = fface_cascade.detectMultiScale(gray, 1.3, 5) #img, scale factor, neighbors
    pface = pface_cascade.detectMultiScale(gray, 1.3, 5)

    # draw rectangles based on frontal face classifier results
    for (x, y, w, h) in fface:
        frect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        print len(frect)
        fface_gray = gray[y:y + h, x:x + w]
        fface_color = image[y:y + h, x:x + w]

    # draw rectangles based on profile face classifier results
    for (x, y, w, h) in pface:
        prect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        print len(prect)
        pface_gray = gray[y:y + h, x:x + w]
        pface_color = image[y:y + h, x:x + w]

    # set the window size, padding around cells, and the scale at which to observe the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.5)# 4,4; 2,2; 1.5
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # for resulting, non-duplicate results
    for (xA, yA, xB, yB) in pick:
        i = 0
        i < 1000
        nRect = [xA, yA, xB, yB]
        nImg = cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    image = imutils.resize(image, width=min(1000, image.shape[1]))
    
    # depending on how you installed ocv/your OS, you may have issues showing/passing/destroying images.
    # saving images out should always work though.

    # cv2.imshow('img', image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    print "reviewed image {}".format(str(j))
    cv2.imwrite("/Users/marioag/Documents/GitHub/gehl-detect/out/find_features/webcam/image{}.png".format(j), image)
    j += 1