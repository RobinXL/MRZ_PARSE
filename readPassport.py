from transform import four_point_transform
import numpy as np
import cv2
import imutils


# 1. run scan to get passport image only
def scan(image_path):
    if not type(image_path) == np.ndarray:
        image = cv2.imread(image_path)
    else:
        image = image_path
    # image = cv2.imread(image_path)
    # Compute the ratio of the old height to the new height, clone it, 
    # and resize it easier for compute and viewing
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    ### convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian Blurring to remove high frequency noise helping in
    # Contour Detection 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    edged = cv2.Canny(gray, 75, 200)
    kernel = np.ones((15,15), np.uint8)  # note this is a horizontal kernel
    d_im = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(d_im, kernel, iterations=1)
    # finding the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Taking only the top 5 contours by Area
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    qualified_cnts = []
    for c in cnts:
        #Calculates a contour perimeter or a curve length
        cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)#0.02
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        screenCnt = approx
        if len(approx) == 4 and cnt_w > 450:
            qualified_cnts.append(approx)
            # screenCnt = approx
            # break

    if qualified_cnts:
        biggest_contour = qualified_cnts[0]
        warped = four_point_transform(orig, biggest_contour.reshape(4, 2) * ratio)
        return warped
    else:
        return orig


# 2. run detect_mrz to get mrz region only
def get_mrz(image_path):
    # load the image, resize it, and convert it to grayscale
    # origin_image = cv2.imread(image_path)
    if not type(image_path) == np.ndarray:
        origin_image = cv2.imread(image_path)
    else:
        origin_image = image_path
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
    # cv2.imwrite('gradX.png', gradX)
    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    # cv2.imwrite('before_thresh.png', thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)
    # cv2.imwrite('thresh.png', thresh)
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
    find_mrz = False
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
            roi = origin_image[int(y*ratio):int((y + h)*ratio), int(x*ratio):int((x + w)*ratio)].copy()
            find_mrz = True
            return roi
    if not find_mrz:
        return origin_image
