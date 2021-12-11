import pytesseract, sys, cv2
# from skimage.filters import threshold_local

cv_im = sys.argv[1]

# T = threshold_local(cv_im, 11, offset = 10, method = "gaussian")
image = cv2.imread(cv_im)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray = cv2.medianBlur(gray, 3)

gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imwrite('beforeOCR.png', gray)

text = pytesseract.image_to_string(gray)

print(text)
