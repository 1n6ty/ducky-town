import numpy as np
from cv2 import cv2

template = cv2.imread('pics/stop_template.jpg')
img = cv2.imread('pics/demo_stop.jpg')

sift = cv2.SIFT_create()

key, des = sift.detectAndCompute(template, None)
key2, des2 = sift.detectAndCompute(img, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des, des2, k=2)
good = [[m] for m, n in matches if m.distance < 0.75*n.distance]

imRes = cv2.drawMatchesKnn(template, key, img, key2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
src_pts = np.float32([ key[m.queryIdx].pt for m, n in matches]).reshape(-1,1,2)
dst_pts = np.float32([ key2[m.trainIdx].pt for m, n in matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = template.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts,M)
dst += (w, 0)

imRes = cv2.polylines(imRes, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
cv2.imshow("win", imRes)
cv2.waitKey(0)