from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-r", "--reference", required=False,
	help="path to reference OCR-A image", default='./ocr-a.jpg')
args = vars(ap.parse_args())


# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = []
ref.append(cv2.imread("1-efitype.jpg"))
ref.append(cv2.imread("2-efitype.jpg"))
ref.append(cv2.imread("3-efitype.jpg"))

for i in range(3):
    ref[i] = cv2.cvtColor(ref[i], cv2.COLOR_BGR2GRAY)
    ref[i] = cv2.threshold(ref[i], 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow("Gray", ref[i])
    cv2.waitKey(0)

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = []
for i in range(3):
    refCnts.append(cv2.findContours(ref[i].copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE))
    refCnts[i] = refCnts[i][0] if imutils.is_cv2() else refCnts[i][1]
    refCnts[i] = contours.sort_contours(refCnts[i], method="left-to-right")[0]

digits = {}

totalRefCnts = []
for i in range(3):
    totalRefCnts.extend(refCnts[i])
# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    cv2.imshow("roi [{}]".format(i), roi)
    cv2.waitKey(0)
    roi = cv2.resize(roi, (57, 88))
	# update the digits dictionary, mapping the digit name to the ROI
    digits[i] = roi
