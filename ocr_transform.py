from transform import four_point_transform
from imutils import contours
from pprint import pprint
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
	image = imutils.resize(image, height = 1000)

	# convert the image to grayscale, blur it, and find edges
	# in the image

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (1, 1), 0)
	edged = cv2.Canny(gray, 75, 200)

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
	help="path to input image", default="ine.jpg")
ap.add_argument("-m", "--manual", required=False,
	help="manually select ROIs", default=False, action="store_true")
args = vars(ap.parse_args())

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
# image = imWarped(image, debug=True)
image = imutils.resize(image, height = 1000)
# image = cv2.GaussianBlur(image, (5, 5), 0)

NAME_ROI = (519, 296, 626, 162)
ADDR_ROI = (510, 489, 794, 181)
CURP_ROI = (605, 715, 467, 71)
BRDY_ROI = (1301, 301, 233, 52)
SEX_ROI = (1373, 355, 176, 59)

if args['manual']:
	fromCenter = False
	NAME_ROI = cv2.selectROI("Name ROI", image, fromCenter)
	ADDR_ROI = cv2.selectROI("ADDR_ROI", image, fromCenter)
	CURP_ROI = cv2.selectROI("CURP_ROI", image, fromCenter)
	BRDY_ROI = cv2.selectROI("BRDY_ROI", image, fromCenter)
	SEX_ROI  = cv2.selectROI("SEX_ROI", image, fromCenter)
	cv2.destroyAllWindows()
	print('NAME_ROI = ' + str(NAME_ROI))
	print('ADDR_ROI = ' + str(ADDR_ROI))
	print('CURP_ROI = ' + str(CURP_ROI))
	print('BRDY_ROI = ' + str(BRDY_ROI))
	print('SEX_ROI = ' + str(SEX_ROI))

locs = [NAME_ROI, ADDR_ROI, CURP_ROI, BRDY_ROI, SEX_ROI]
locs = sorted(locs, key=lambda x:x[1])
"""
Each rectangle (area of text)
"""
for (i, (gX, gY, gW, gH)) in enumerate(locs):
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
	cv2.imwrite(filename, mask)

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
