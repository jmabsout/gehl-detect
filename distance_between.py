# import the necessary packages

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from imutils.object_detection import non_max_suppression
import math
import glob
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


img_dir = "t4"
img_t = "/Users/Mario/Documents/mit-github-projects/gehl/Gehl/t3old/G0082212_red2.jpg"
face_cascade = cv2.CascadeClassifier("/Users/Mario/Documents/mit-github-projects/gehl/Gehl/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt2.xml")


# defines images in a directory
images = []
os.chdir(img_dir)
for file in glob.glob("*.JPG"):
    images.append(file)

# defines reference object parameters in reference image
refObj = [708,1089,756,1137]
refObjMid = ((((refObj[2]-refObj[0])/2)+refObj[0]), (((refObj[3]- refObj[1])/2)+refObj[1]))
refWidth = refObj[2] - refObj[0]

KNOWN_DISTANCE_f = 96.0
KNOWN_WIDTH_f = 5.0
KNOWN_DISTANCE_r = 149.0
KNOWN_WIDTH_r = 5.75


# sets up reference object based on image with known parameters
image = cv2.imread(img_t)

# generates image statistics
height, width, channels = np.shape(image)
# print width
img_center = width/2
img_cent_orig = (img_center, height)
angle = refObjMid[0]*90/img_center

# print angle

# this function actually does the detection / classification
def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print type(face)
    if type(face) is tuple:
        #TODO: find a better way to deal with negative images
        rects = [[0,0,1,1], "noface"]
        # print "no face"
        rectsMid = [0.5,0.5]
        return rects, rectsMid
    else:
        for (x, y, w, h) in face:
            i = 0
            rect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])
            rect2 = cv2.rectangle(image,(refObj[0], refObj[1]),(refObj[2],refObj[3]),(0,0,255),2)
            # print "good face rects =  ",rects
            rectsMid = ((((rects[0][2]-rects[0][0])/2)+rects[0][0]),(((rects[0][3]-rects[0][1])/2)+rects[0][1]))
            # rs = rects[0]
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            distLine = cv2.line(image, refObjMid, rectsMid, (0, 0, 255), 5)
            camFaceLine = cv2.line(image, img_cent_orig, rectsMid, (0, 0, 255), 5)
            camRefLine = cv2.line(image, img_cent_orig, refObjMid, (0, 0, 255), 5)
            i +=1

            return rects, rectsMid


camPos = (766003.20, 2955893.08)
refObjPos = (765996.22, 2955900.69)

# compute and return the distance from the maker to the camera
def distance_to_camera(knownWidth, focalLength, perWidth):

    return (knownWidth * focalLength) / perWidth

# translates distance observed into triangles used to find x,y position in plotting
def get_coord_pair(dist_f, rectsMid, dist_r):
    camCoord = (img_center, height)

    # finds sides of ref obj triangle in pixels
    refObjMidx = (img_center - refObjMid[0])
    refObjMidy = (height - refObjMid[1])
    distPix_r = math.sqrt(refObjMidx**2+refObjMidy**2)

    # finds scale factor to multiply pixel sides by
    scale_r = dist_r / distPix_r
    refObjMidx_feet = scale_r * refObjMidx
    refObjMidy_feet = scale_r * refObjMidy
    refObj_a_b_c = refObjMidx_feet, refObjMidy_feet, dist_r

    # finds sides of face obj triangle in pixels
    faceObjMidx = -(img_center - rectsMid[0])
    print "faceobjmidx, pixels: ", faceObjMidx
    faceObjMidy = (height - rectsMid[1])
    distPix_f = math.sqrt(faceObjMidx ** 2 + faceObjMidy ** 2)

    # finds scale factor to multiply pixel sides by
    scale_f = dist_f / distPix_f
    print "scale_f: ", scale_f
    faceObjMidx_feet = scale_f * faceObjMidx
    print "faceobjmidx, feet: ", faceObjMidx_feet
    faceObjMidy_feet = scale_f * faceObjMidy
    faceObj_a_b_c = faceObjMidx_feet, faceObjMidy_feet, dist_f

    refObjCoord = ((img_center- refObjMid[0]), feet_r) # right now x is in pixels and y is in feet...
    objCoord = ((img_center- rectsMid[0]), feet_f) # right now x is in pixels and y is in feet...

    return camCoord, refObjCoord, objCoord, refObj_a_b_c, faceObj_a_b_c


# sets up reference image

# finds marker in reference image
marker, midpoint = find_marker(image)
faceWidth = (marker[0][2]-marker[0][0])

# finds focal length information in reference image
focalLength_f = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r
print "focallength f:   ", focalLength_f
# prints distance between in reference image
faceWidth = (marker[0][2] - marker[0][0])
inches_f = distance_to_camera(KNOWN_WIDTH_f, focalLength_f, faceWidth)
inches_r = distance_to_camera(KNOWN_WIDTH_r, focalLength_r, refWidth)
feet_f = inches_f/12
feet_r = inches_r/12
# print "distance to face: ", feet_f, "feet."
# print "distance to refobj: ", feet_r, "feet."

distBetween = math.sqrt(feet_f*feet_f + feet_r*feet_r - (2*feet_f*feet_r*math.cos(angle)))
# print "computed distbetween: ", distBetween
# print "actual distbetween (measured): 5.916667"

# sets up lists to facilitate plotting below
faceX = []
faceY = []
j = []

i=1

# applies information gleaned from reference image and applies to all images in directory
for each in images:

    image = cv2.imread(each)
    marker, midpoint = find_marker(image)
    print "marker: ", marker
    faceWidth = (marker[0][2]-marker[0][0])
    focalLength = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
    focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r
    print "flf: ", focalLength_f
    print "kwf: ", KNOWN_WIDTH_f
    print "kdf: ", KNOWN_DISTANCE_f
    inches_f = distance_to_camera(KNOWN_WIDTH_f, focalLength_f, faceWidth)
    print "if: ", inches_f
    inches_r = distance_to_camera(KNOWN_WIDTH_r, focalLength_r, refWidth)
    feet_f = inches_f/12
    feet_r = inches_r/12

    distBetween = math.sqrt(feet_f*feet_f + feet_r*feet_r - (2*feet_f*feet_r*math.cos(angle)))

    cam, ref, face, refTriangle, faceTriangle = get_coord_pair(feet_f, midpoint, feet_r)

    faceX.append(int(faceTriangle[0]))
    faceY.append(int(faceTriangle[1]))
    print "x: ",faceX, "y: ", faceY

    print "reference triangle side lengths (feet): ",refTriangle
    print "face triangle side lengths (feet): ", faceTriangle
    print "distance to camera: ", feet_f
    print "computed distbetween: ", distBetween
    # print "actual distbetween (measured): 5.916667"
    j.append(i)

    # styles plot
    plt.axis([-10,10,0,30])
    plt.set_cmap("Reds")
    numImg = len(images)
    c = [float(i)/float(numImg), 0.0, float(numImg-i)/float(numImg)]
    plot = plt.scatter(faceX, faceY, s = 100, c=j, edgecolors='black')

    # adds grid to plot
    # plt.grid()

    # plt.show()
    plt.savefig("cumu_image{}.png".format(i))
    i += 1
    cv2.putText(image, "fname %s" % (str(each)),
                (image.shape[1] - 600, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)
    # resizes images so it's easier to see on screen, shows image
    img = imutils.resize(image, width=min(1000, image.shape[1]))
    cv2.imshow("image", img)

    #saves image to disk
    # cv2.imwrite("image{}.png".format(i), img)
    cv2.waitKey(1)