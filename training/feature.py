import numpy as np
import cv2

BIN_NUM = 16

def hog(image_cell):
    # image_cell size is 20*20
    gx = cv2.Sobel(image_cell, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image_cell, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(BIN_NUM * ang/(2*np.pi))

    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), BIN_NUM) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # hist is 64 bit vector
    return hist

def extract_feature(image):
    cells = [np.hsplit(row, 7) for row in np.vsplit(image, 10)]
    hog_data = [map(hog, row) for row in cells]
    return np.float32(hog_data).ravel()