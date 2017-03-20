import os

import cv2
import numpy as np

import feature

BIN_NUM = 16

SVM_PARAMS = dict(kernel_type=cv2.ml.SVM_LINEAR,
                  svm_type=cv2.ml.SVM_C_SVC,
                  C=2.67, gamma=5.383)

SIGN_NEGATIVE = -1
SIGN_POSITIVE = 1
SIGN_LIST = [SIGN_NEGATIVE, SIGN_POSITIVE]

POSITIVE_SAMPLE_IMAGE_PATH = 'INRIAPerson/Train/pos'
NEGATIVE_SAMPLE_IMAGE_PATH = 'INRIAPerson/Train/neg'


# Color information is useless when processing
# So image is converted to gray space before yielding.
def load_sample_image(sign):
    if sign == SIGN_POSITIVE:
        path = POSITIVE_SAMPLE_IMAGE_PATH
    elif sign == SIGN_NEGATIVE:
        path = NEGATIVE_SAMPLE_IMAGE_PATH

    for (dir_name, dir_list, file_list) in os.walk(path):
        for file in file_list:
            image_name = dir_name + '/' + file
            # Generally, here only contains jpeg files but
            # we still need to check in case of the occurance
            # of .spotlight and other system generated file
            if image_name.endswith('.png'):
                image = cv2.imread(image_name)
                # print image_name
                # Convert to gray space
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                yield image


def main():
    # Sliding window size is 200*140(h*w)
    train_list = []
    response_list = []

    # Process sample images
    for SIGN in SIGN_LIST:
        for sample_image in load_sample_image(SIGN):
            train_data = feature.extract_feature(sample_image)
            train_list.append(train_data)
            response_list.append(SIGN)

    # SVM in OpenCV 3.1.0 for Python
    SVM = cv2.ml.SVM_create()
    SVM.setKernel(cv2.ml.SVM_LINEAR)
    SVM.setP(0.2)
    SVM.setType(cv2.ml.SVM_EPS_SVR)
    SVM.setC(1.0)

    tl = np.array(train_list, np.float32)
    rl = np.array(response_list, np.int32)
    print tl
    # print rl
    # Train SVM model
    # svm = cv2.SVM()

    SVM.train(tl, cv2.ml.ROW_SAMPLE, rl)
    SVM.save('svm_data.dat')

    return


if __name__ == '__main__':
    main()