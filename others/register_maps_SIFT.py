from __future__ import print_function

import cv2
import numpy as np

import matplotlib.transforms
import matplotlib.pyplot as plt

################################################################################
################################################################# initialization
################################################################################
img_src_dis = cv2.imread('/home/saesha/Dropbox/myGits/sample_data/HH/HIH/HIH_04_dis.png', cv2.IMREAD_GRAYSCALE)
img_src_dis = cv2.imread('/home/saesha/Documents/tango/HIH_03/20170409123544_dis.png', cv2.IMREAD_GRAYSCALE)
img_dst_dis = cv2.imread('/home/saesha/Documents/tango/HIH_02/20170409123351_dis.png', cv2.IMREAD_GRAYSCALE)

img_src = cv2.imread('/home/saesha/Dropbox/myGits/sample_data/HH/HIH/HIH_04.png', cv2.IMREAD_GRAYSCALE)
img_src = cv2.imread('/home/saesha/Documents/tango/HIH_03/20170409123544.png', cv2.IMREAD_GRAYSCALE)
img_dst = cv2.imread('/home/saesha/Documents/tango/HIH_02/20170409123351.png', cv2.IMREAD_GRAYSCALE)

################################################################################
####################################################################### dev yard
################################################################################
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html


#################### Brute-Force Matching with ORB Descriptors
img1_dis = img_src_dis.copy()
img2_dis = img_dst_dis.copy()

img1 = img_src.copy()
img2 = img_dst.copy()


# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1_dis,None)
kp2, des2 = orb.detectAndCompute(img2_dis,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], outImg=None, flags=2)

plt.imshow(img3),plt.show()


#################### Brute-Force Matching with SIFT Descriptors and Ratio Test
img1_dis = img_src_dis.copy()
img2_dis = img_dst_dis.copy()

img1 = img_src.copy()
img2 = img_dst.copy()

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_dis,None)
kp2, des2 = sift.detectAndCompute(img2_dis,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, outImg=None, flags=2)

plt.imshow(img3),plt.show()


#################### FLANN based Matcher
img1_dis = img_src_dis.copy()
img2_dis = img_dst_dis.copy()

img1 = img_src.copy()
img2 = img_dst.copy()

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1_dis,None)
kp2, des2 = sift.detectAndCompute(img2_dis,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()

