from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=False,
	help="path to reference OCR-A image")
args = vars(ap.parse_args())


# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread(args["reference"])
ref = imutils.resize(ref, width=100)

ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]


# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="top-to-bottom")[0]
digits = {}


# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi


# initialize a rectangular (wider than it is tall) and square
# structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image", gray)
#cv2.waitKey(0)


# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the text areas)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
#cv2.imshow("Tophat", tophat)
#cv2.waitKey(0)

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
#cv2.imshow("Thresh", thresh)
#cv2.waitKey(0)


# apply a second closing operation to the binary image, again
# to help close gaps between credit card number regions
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv2.imshow("Thresh-2", thresh)
# cv2.waitKey(0)

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


# loop over the 4 groupings of 4 digits
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
    cv2.imshow("group threshed" + str(i), group)
    cv2.waitKey(0)


    """
    Each letter for every section
    """
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_LIST,
    	cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]

    # Get Contours that could be letters based on size
    actualLetters = []
    for (index, possibleLetter) in enumerate(digitCnts):
        (x, y, w, h) = cv2.boundingRect(possibleLetter)
        if (h > 5):
            actualLetters.append(possibleLetter)

    digitCnts = contours.sort_contours(actualLetters,
    	method="left-to-right")[0]

    # Try to match the letter to the template.
    for c in digitCnts:
        # Bounding Rectangle of an area found in the ID.
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        #cv2.imshow("roi :" + str(c), roi)
        #cv2.waitKey(0)

        roi = cv2.resize(roi, (57, 88))

        #cv2.imshow("roi :" + str(c), roi)
        #cv2.waitKey(0)

        # initialize a list of template matching scores
        scores = []


        # loop over the reference name and ROI
        for (digit, digitROI) in digits.items():
        	# apply correlation-based template matching, take the
        	# score, and update the scores list
        	result = cv2.matchTemplate(roi, digitROI,
        		cv2.TM_CCOEFF)
        	(_, score, _, _) = cv2.minMaxLoc(result)
        	scores.append(score)

        # the classification for the digit ROI will be the reference
        # digit name with the *largest* template matching score
        groupOutput.append(str(np.argmax(scores)))
    """END
    Each letter for every section
    """

    # draw the digit classifications around the group
    cv2.rectangle(image, (gX - 5, gY - 5),
    	(gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
    	cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # update the output digits list
    output.extend(groupOutput)


# display the output credit card information to the screen
print("OCR: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
