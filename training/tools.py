import imutils


def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def image_pyramid_scale(image, scale=1.25, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid and yield the resized image
    while True:
        scaled_width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=scaled_width)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image