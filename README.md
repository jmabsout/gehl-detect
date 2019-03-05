# Gehl

## Getting started
Install OpenCV via build, as per: http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/

Or, conversely, use `python pip -r requirements.txt` from within this directory (although you may be unable to show images in an IDE if you don't build from source).

`find_features.py` should work with any directory of images that you pass it to demonstrate OpenCV's classifiers. `measure_distance.py` requires a bit more setup, with a reference image and reference object within that image to triangulate position of faces detected relative to the camera. The examples in examples/distance/out/ show the result of this file.
