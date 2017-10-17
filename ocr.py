
from imutils import contours
from PIL import Image
import pytesseract
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the text areas)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv2.imshow("Tophat", tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# compute the Scharr gradient of the tophat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")


# apply a closing operation using the rectangular kernel to help
# cloes gaps in between letters, then apply
# Otsu's thresholding method to binarize the image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply a second closing operation to the binary image, again
# to help close gaps.
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv2.imshow("Thresh-2", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours in the thresholded image, then initialize the
# list of text locations
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
locs = []


# loop over the contours
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)

    if (w > 20) and (h > 10 and h < 30):
        locs.append((x, y, w, h))


# sort the areas of text from top-to-bottom, then initialize the
# list of classified text
locs = sorted(locs, key=lambda x:x[1])
output = []


"""
Each rectangle (area of text)
"""
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
	groupOutput = []

    # Extract the rectangle from the gray image
	group = image[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	hsv = cv2.cvtColor(group, cv2.COLOR_BGR2HSV)
	cv2.imshow("field", group)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow("hsv", hsv)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	lower_black = np.array([0, 0, 0])
	upper_black = np.array([180, 255, 40])

	mask = cv2.inRange(hsv, lower_black, upper_black)
	# dilate the mask image
	kernel = np.ones((1, 1), np.uint8)
	mask = cv2.erode(mask, kernel, iterations=1)
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.dilate(mask, kernel, iterations=1)
	cv2.imshow('mask', mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	res = cv2.bitwise_and(hsv, hsv, mask=mask)
	cv2.imshow('res', res)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

    # write the image to disk as a temporary file so we can
    # apply OCR to it
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, group)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
	text = pytesseract.image_to_string(Image.open(filename), lang='spa.news-gothic-light')
	text_default = pytesseract.image_to_string(Image.open(filename), lang='spa')
	text_eng = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	print("text spa: %s \ntext spa.news-gothic: %s \ntext eng: %s\n" % (text_default, text, text_eng))
	cv2.imshow("group threshed" + str(i), mask)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# display the output ID information to the screen
# print("OCR: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
