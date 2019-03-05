# a file to refactor distance_between.py and gehl_detect.py; goal is to move most functions to gehl_detect.py and have this file primarily hold parameter definitions
'''
This file demonstrates how you can identify persons and calculate their position relative to the camera and a reference object.
This assumes you have a reference object with a face a known distance from the camera, and you know the position of a reference
point within the image. The examples/measure_disatance/in contains images that fit this description, and the out/ directory
shows the results, both as images and x,y plots that show position relative to the camera.
'''


# import libraries
import numpy as np
import imutils
import cv2
from imutils.object_detection import non_max_suppression
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gehl_detect as gd

# parameter definitions

img_dir = '/Users/marioag/Dropbox (Personal)/documents/mit-github-projects/gehl/Gehl/t4'
reference_img = '/Users/marioag/Dropbox (Personal)/documents/mit-github-projects/gehl/Gehl/t3/G0082212.JPG'
face_cascade = cv2.CascadeClassifier('/Users/marioag/Dropbox (Personal)/documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml')

images = gd.get_jpgs(img_dir)
# print images

# bounding box of reference object in reference image
refObj = [708,1089,756,1137]
refObjMid, refWidth = gd.get_refObj_statistics(refObj)

# kwown distance/width of face, reference object in reference image
KNOWN_DISTANCE_f = 96.0
KNOWN_WIDTH_f = 5.0
KNOWN_DISTANCE_r = 149.0
KNOWN_WIDTH_r = 5.75

# lat/lon of camera and reference object in space
camPos = (766003.20, 2955893.08)
refObjPos = (765996.22, 2955900.69)

# define reference image statistics
image = cv2.imread(reference_img)

height, width, img_center, img_cent_orig = gd.get_image_statistics(image)

# get angle between midpoint of reference object and center of image
angle = refObjMid[0]*90/img_center

# finds marker in reference image
marker, midpoint = gd.find_faces(image, face_cascade, refObj, refObjMid, img_cent_orig)

faceWidth = gd.get_face_width(marker)
# print "facewidth:  ", faceWidth

focalLength_f, focalLength_r = gd.get_focal_length(faceWidth, refWidth, KNOWN_DISTANCE_r, KNOWN_DISTANCE_f, KNOWN_WIDTH_f, KNOWN_WIDTH_r)

# calculates distance between face, refobj, and camera in reference image
inches_f, inches_r, feet_f, feet_r, distBetween = gd.get_distance(faceWidth, refWidth, angle, KNOWN_WIDTH_f, KNOWN_WIDTH_r, focalLength_f, focalLength_r)

faceX = []
faceY = []
j = []

i = 0

for each in images:
    print "processing image ", each
    img = cv2.imread(each)
    # height, width, img_center, img_cent_orig = gd.get_image_statistics(image)
    marker, midpoint = gd.find_faces(img, face_cascade, refObj, refObjMid, img_cent_orig)
    faceWidth = gd.get_face_width(marker)
    # print "marker: ", marker, "  midpoint: ", midpoint
    inches_f, inches_r, feet_f, feet_r, distBetween = gd.get_distance(faceWidth, refWidth, angle, KNOWN_WIDTH_f, KNOWN_WIDTH_r, focalLength_f, focalLength_r)
    # print "inchesF:  ", inches_f
    # print faceWidth, focalLength_f, focalLength_r, inches_f

    # this block calculate the distance between the detected faces and the camera; you can print or print to file
    # faceWidth = (marker[0][2] - marker[0][0])
    # focalLength = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
    # focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r
    # print "flf: ", focalLength_f
    # print "kwf: ", KNOWN_WIDTH_f
    # print "kdf: ", KNOWN_DISTANCE_f
    # inches_f = gd.distance_to_camera(KNOWN_WIDTH_f, focalLength_f, faceWidth)
    # print "if: ", inches_f
    # inches_r = gd.distance_to_camera(KNOWN_WIDTH_r, focalLength_r, refWidth)
    # feet_f = inches_f / 12
    # feet_r = inches_r / 12
    #
    # distBetween = math.sqrt(feet_f * feet_f + feet_r * feet_r - (2 * feet_f * feet_r * math.cos(angle)))

    cam, ref, face, refTriangle, faceTriangle = gd.get_coord_pair(feet_f, midpoint, feet_r, img_center, height, refObjMid)

    faceX.append(int(faceTriangle[0]))
    faceY.append(int(faceTriangle[1]))
    # print "x: ", faceX, "y: ", faceY

    print "reference triangle side lengths (feet): ", refTriangle
    print "face triangle side lengths (feet): ", faceTriangle
    print "distance to camera: ", feet_f
    print "computed distance between: ", distBetween
    # print "actual distbetween (measured): 5.916667"
    j.append(i)

    # styles plot
    plt.axis([-10, 10, 0, 30])
    plt.set_cmap("Reds")
    numImg = len(images)
    c = [float(i) / float(numImg), 0.0, float(numImg - i) / float(numImg)]
    plot = plt.scatter(faceX, faceY, s=100, c=j, edgecolors='black')

    # adds grid to plot
    # plt.grid()

    # plt.show()
    # outputs the accompanying location plot; this assumes you have properly supplied
    # a reference image and a reference object within that image.

    plt.savefig("/Users/marioag/Documents/GitHub/gehl-detect/out/distance/plot{}.png".format(i))
    i += 1
    print i
    # resizes images so it's easier to see on screen, shows image
    cv2.putText(img, "fname %s" % (str(each)),
                (img.shape[1] - 600, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)
    img = imutils.resize(img, width=min(2000, img.shape[1]))

    # as mentioned in find_features.py, your os/the way you installed ocv may preclude
    # your ability to show/destroy/pass on images. saving should always work though.
        
    # cv2.imshow("image", img)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # saves image to disk
    cv2.imwrite("/Users/marioag/Documents/GitHub/gehl-detect/out/distance/image{}.png".format(i), img)
    

    