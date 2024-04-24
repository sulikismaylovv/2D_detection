import cv2
import numpy as np

# Load the image
img = cv2.imread('testV2/test14.jpg')

# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 25, 192)
# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)
thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
thresh = cv2.dilate(thresh,None,iterations =5)
thresh = cv2.erode(thresh,None,iterations =5)

# Find the contours
contours,hierarchy = cv2.findContours(thresh,
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area >75000 and area < 250000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,
                  (x,y),(x+w,y+h),
                  (0,255,0),
                  5)



cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()