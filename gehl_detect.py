# python file containing functions for ped detection, image processing, etc.
# meant to serve as toolkit for all methods and scripts written thus far

import numpy as np
import imutils
import cv2
from imutils.object_detection import non_max_suppression
import math
import glob
import os
import matplotlib
matplotlib.use("TkAgg")  # this is specifically for plotting with matplotlib on macosx el capitan
from matplotlib import pyplot as plt

# gets jpegs out of a directory
def get_jpgs(imgDir):
    images = []
    os.chdir(imgDir)
    for file in os.listdir(imgDir):
        # print file
        if file.lower().endswith(".jpg"):
            images.append(file)

    return images

# gets pngs out of a directory
def get_pngs(imgDir):
    images = []
    os.chdir(imgDir)
    for file in os.listdir(imgDir):
        if file.lower().endswith(".png"):
            images.append(file)
    return images

def get_refObj_statistics(refObj):
    refObjMid = ((((refObj[2] - refObj[0]) / 2) + refObj[0]), (((refObj[3] - refObj[1]) / 2) + refObj[1]))
    refWidth = refObj[2] - refObj[0]

    return refObjMid, refWidth

def get_image_statistics(image):
    height, width, channels = np.shape(image)
    img_center = width / 2
    img_cent_orig = (img_center, height)
    return height, width, img_center, img_cent_orig

def distance_to_camera(knownWidth, focalLength, perWidth):
    # print "kw_dtc: ", knownWidth
    # print "fl: ", focalLength
    print "pw: ", perWidth
    dtc = (knownWidth * focalLength) / perWidth
    print "DTC:   ", dtc

    return (knownWidth * focalLength) / perWidth

def find_faces(image, classifier, refObj, refObjMid, img_cent_orig):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = classifier.detectMultiScale(gray, 1.3, 5)
    if type(face) is tuple:
        # TODO: find a better way to deal with negative images
        rects = [[0, 0, 1, 1], "noface"]
        # print "no face"
        rectsMid = [0.5, 0.5]
        return rects, rectsMid
    else:
        for (x, y, w, h) in face:
            i = 0
            rect = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face])
            rect2 = cv2.rectangle(image, (refObj[0], refObj[1]), (refObj[2], refObj[3]), (0, 0, 255), 2)
            # print "good face rects =  ",rects
            rectsMid = (
            (((rects[0][2] - rects[0][0]) / 2) + rects[0][0]), (((rects[0][3] - rects[0][1]) / 2) + rects[0][1]))
            # rs = rects[0]
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            distLine = cv2.line(image, refObjMid, rectsMid, (0, 0, 255), 5)
            camFaceLine = cv2.line(image, img_cent_orig, rectsMid, (0, 0, 255), 5)
            camRefLine = cv2.line(image, img_cent_orig, refObjMid, (0, 0, 255), 5)
            i += 1

            return rects, rectsMid

def get_focal_length(faceWidth, refWidth, KNOWN_DISTANCE_r, KNOWN_DISTANCE_f, KNOWN_WIDTH_f, KNOWN_WIDTH_r):
    focalLength_f = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
    focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r

    return focalLength_f, focalLength_r

def get_face_width(marker):
    faceWidth = (marker[0][2] - marker[0][0])

    return faceWidth

def get_distance(faceWidth, refWidth, angle, KNOWN_WIDTH_f, KNOWN_WIDTH_r, focalLength_f, focalLength_r):
    # finds marker in reference image
    # faceWidth = (marker[0][2] - marker[0][0])
    # print "fwgd: ", faceWidth
    # finds focal length information in reference image
    # focalLength_f = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
    # focalLength_f = 1804.8
    # focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r
    # print "flf: ", focalLength_f
    # print "kwf: ", KNOWN_WIDTH_f
    # print "kdf: ", KNOWN_DISTANCE_f
    # prints distance between in reference image
    # faceWidth = (marker[0][2] - marker[0][0])
    inches_f = distance_to_camera(KNOWN_WIDTH_f, focalLength_f, faceWidth)
    # print "if: ", inches_f
    inches_r = distance_to_camera(KNOWN_WIDTH_r, focalLength_r, refWidth)
    feet_f = inches_f / 12
    # print "feetf: ", feet_f
    feet_r = inches_r / 12

    distBetween = math.sqrt(feet_f ** 2 + feet_r ** 2 - (2 * feet_f * feet_r * math.cos(angle)))
    # print "DISTANCE BETWEEN:  ",distBetween
    return inches_f, inches_r, feet_f, feet_r, distBetween

def get_coord_pair(dist_f, rectsMid, dist_r, img_center, height, refObjMid):

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

    refObjCoord = ((img_center- refObjMid[0]), dist_r)
    objCoord = ((img_center- rectsMid[0]), dist_f)

    return camCoord, refObjCoord, objCoord, refObj_a_b_c, faceObj_a_b_c