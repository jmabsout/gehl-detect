import tools
import cv2
import imutils
import os

SLICED_IMAGE_BASE_PATH = './slicedimage/'
RAW_IMAGE_BASE_PATH = './rawpics/'


def slice_image(image, sliced_image_size, step_size):
    for resized in tools.image_pyramid_scale(image, scale=1.25):
        for (x, y, sliced_image) in tools.sliding_window(resized, stepSize=step_size, windowSize=sliced_image_size):
            # This line is dirty, rewrite it later
            if sliced_image.shape[0] != sliced_image_size[1] or sliced_image.shape[1] != sliced_image_size[0]:
                continue

            yield sliced_image


def load_image():
    for (dir_path, dir_list, file_list) in os.walk(RAW_IMAGE_BASE_PATH):
        for file in file_list:
            if file.endswith('.jpg'):
                image_name = dir_path + '/' + file
                image = cv2.imread(image_name)
                try:
                    if image.size:
                        yield (image, file)
                except AttributeError:
                    print 'warning: ' + image_name + ' not processed'


def preprocess(image, image_name, fixed_size):
    # Make a new directory for the image according to its name
    path = SLICED_IMAGE_BASE_PATH + image_name + '/'
    os.mkdir(path)

    # Set the image to fixed size
    image = imutils.resize(image, width=fixed_size[0], height=fixed_size[1])

    return image, path


def main():
    for image, image_name in load_image():
        image, path = preprocess(image, image_name, [600, 800])
        for i, sliced_image in enumerate(slice_image(image, [140, 200], 20)):
            sliced_image_path = path + str(i) + '.jpg'
            cv2.imwrite(sliced_image_path, sliced_image)
    return


if __name__ == '__main__':
    main()