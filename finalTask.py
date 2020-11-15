import numpy as np
from cv2 import cv2

img = cv2.imread('pics/final_challenge.png')
template = cv2.imread('pics/index.png')
cv2.namedWindow('img')
def nothing(x): pass
cv2.createTrackbar('H', 'img', 216, 255, nothing)
cv2.createTrackbar('S', 'img', 0, 255, nothing)
cv2.createTrackbar('V', 'img', 0, 255, nothing)

cv2.createTrackbar('HH', 'img', 255, 255, nothing)
cv2.createTrackbar('SH', 'img', 255, 255, nothing)
cv2.createTrackbar('VH', 'img', 180, 255, nothing)

low = np.array([cv2.getTrackbarPos('H', 'img'), cv2.getTrackbarPos('S', 'img'), cv2.getTrackbarPos('V', 'img')])
high = np.array([cv2.getTrackbarPos('HH', 'img'), cv2.getTrackbarPos('SH', 'img'), cv2.getTrackbarPos('VH', 'img')])
mask = cv2.inRange(img, low, high)
contour, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
points = cv2.boundingRect(contour[0])
cv2.rectangle(img, (points[0], points[1]), (points[0] + points[1], points[1] + points[3]), (0,255,0),2)

cv2.imshow("win", img)

cv2.waitKey(0)

cv2.destroyAllWindows()