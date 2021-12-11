# import the necessary packages
# from imutils import paths
import numpy as np
# import argparse
import imutils
import cv2, os, sys
# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())
# initialize a rectangular and square structuring kernel

imagePath = sys.argv[1]
# load the image, resize it, and convert it to grayscale
origin_image = cv2.imread(imagePath)
print('image size: ', origin_image.shape)
o_h, o_w = origin_image.shape[:2]
if o_h > o_w:
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
else:
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))


image = imutils.resize(origin_image, height=600)
n_h, n_w = image.shape[:2]
ratio = o_h / n_h
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# smooth the image using a 3x3 Gaussian, then apply the blackhat
# morphological operator to find dark regions on a light background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
cv2.imwrite('gradX.png', gradX)
# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
cv2.imwrite('before_thresh.png', thresh)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=4)
cv2.imwrite('thresh.png', thresh)
# during thresholding, it's possible that border pixels were
# included in the thresholding, so let's set 5% of the left and
# right borders to zero
p = int(image.shape[1] * 0.05)
thresh[:, 0:p] = 0
thresh[:, image.shape[1] - p:] = 0
# find contours in the thresholded image and sort them by their
# size
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and use the contour to
    # compute the aspect ratio and coverage ratio of the bounding box
    # width to the width of the image
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    crWidth = w / float(gray.shape[1])
    # check to see if the aspect ratio and coverage width are within
    # acceptable criteria
    if ar > 5 and crWidth > 0.75:
        # pad the bounding box since we applied erosions and now need
        # to re-grow it
        pX = int((x + w) * 0.03)
        pY = int((y + h) * 0.03)
        (x, y) = (x - pX, y - pY)
        (w, h) = (w + (pX * 2), h + (pY * 2))
        # extract the ROI from the image and draw a bounding box
        # surrounding the MRZ
        print('ratio: ', ratio)
        roi = origin_image[int(y*ratio):int((y + h)*ratio), int(x*ratio):int((x + w)*ratio)].copy()
        cv2.imwrite('mrz.png', roi)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break
cv2.imwrite('res.png', image)