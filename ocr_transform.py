from transform import four_point_transform
from imutils import contours
from PIL import Image
import pytesseract
import numpy as np
import argparse
import imutils
import cv2
import os

def imWarped(image, debug=False):
	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges
	# in the image

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	edged = cv2.Canny(thresh, 75, 200)

	# show the original image and the edge detected image
	if debug:
		print("STEP 1: Edge Detection")
		cv2.imshow("Image", image)
		cv2.imshow("Edged", edged)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	_, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# show the contour (outline) of the piece of paper
	if debug:
		print("STEP 2: Find contours of paper")
		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Outline", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect

	# show the original and scanned images
	if debug:
		print("STEP 3: Apply perspective transform")
		cv2.destroyAllWindows()
		cv2.imshow("Scanned", imutils.resize(warped, height = 650))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return warped

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
image = imWarped(image, debug=True)
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the text areas)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv2.imshow("Tophat", tophat)
cv2.waitKey(0)

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


# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv2.imshow("Thresh-2", thresh)
# cv2.waitKey(0)

NAME_ROI = (186, 108, 255, 56)
ADDR_ROI = (182, 179, 281, 61)
CURP_ROI = (226, 265, 170, 20)
BRDY_ROI = (487, 106, 81, 16)
SEX_ROI = (524, 122, 64, 26)

# Uncomment to recalculate ROIs from ID
# fromCenter = False
# NAME_ROI = cv2.selectROI("Image", tophat, fromCenter)
# ADDR_ROI = cv2.selectROI("Image", tophat, fromCenter)
# CURP_ROI = cv2.selectROI("Image", tophat, fromCenter)
# BRDY_ROI = cv2.selectROI("Image", tophat, fromCenter)
# SEX_ROI  = cv2.selectROI("Image", tophat, fromCenter)
# print('NAME_ROI = ' + str(NAME_ROI))
# print('ADDR_ROI = ' + str(ADDR_ROI))
# print('CURP_ROI = ' + str(CURP_ROI))
# print('BRDY_ROI = ' + str(BRDY_ROI))
# print('SEX_ROI = ' + str(SEX_ROI))

locs = [NAME_ROI, ADDR_ROI, CURP_ROI, BRDY_ROI, SEX_ROI]
locs = sorted(locs, key=lambda x:x[1])
"""
Each rectangle (area of text)
"""
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # initialize the list of group digits
    groupOutput = []

    # Extract the rectangle from the gray image
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    #cv2.imshow("group" + str(i), group)
    #cv2.waitKey(0)
    group = cv2.threshold(group, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

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
    cv2.imshow("group threshed" + str(i), group)
    cv2.waitKey(0)

# display the output credit card information to the screen
# print("OCR: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
