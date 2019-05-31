import numpy as np
import cv2
from matplotlib import pyplot as plt
import arf  
from clairvoyance import img_tools 

#from hist_match import hist_match
#matched = hist_match(source, template)


file2 = r'D:\Frames\frames-TM1490202529GR00-4\Frames-tm1490202529gr00-4.arf'
 
arf_obj = arf.read(file2, 'r') 
img1 = arf_obj.load(5) 
img1 = img1 [540:592,888:952]

#img1 = img1 [230:323,501:562]


img1 = img_tools.scale_frame_by_percentile(img1, low_pct=1, high_pct=99)*255
img1 = img1.astype(np.uint8)
plt.imshow(img1, cmap = 'gray')
#img2 = img2.astype(np.uint8)
img2 = arf_obj.load(2) 
img2 = img_tools.scale_frame_by_percentile(img2,low_pct=1, high_pct=99)*255
img2 = img2.astype(np.uint8)
#img2 = img_tools.scale_frame_by_percentile(img2,low_pct=1, high_pct=99)
#img1 = img21 [542:590,890:950]
#img1 = cv2.imread('box.png',0)     # queryImag
#img2 = cv2.imread('ref.png',0) # trainImage
#

# Initiate detector
orb = cv2.ORB_create()
surf = cv2.xfeatures2d.SURF_create(4)
#sift = cv2.SIFT()
kaze =cv2.KAZE_create()
brisk = cv2.BRISK_create()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 =sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
#bf = cv2.BFMatcher()
matches = flann.knnMatch(des1,des2,k=2)
# create BFMatcher object
#bf = cv2.BFMatcher()#cv2.NORM_L2, crossCheck=False)
#
## Match descriptors.
#matches = bf.knnMatch(des1,des2, k=2)#bf.match(des1,des2)
#
## Sort them in the order of their distance.
#matches = sorted(matches, key = lambda m:m.distance)

# Apply ratio test
good = []
good_no_list =[]
for m,n in matches:
    if m.distance < 0.96*n.distance:
        good.append([m])
        good_no_list.append(m)
# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)

plt.imshow(img3),plt.show()
#plt.imsave('sift_KNN2_096.jpg',img3)


pts_frame = ([kp2[idx].pt for idx in range(0, len(kp2))])
pts_obj = ([kp1[idx].pt for idx in range(0, len(kp1))])


height, width = img2.shape
blk_step = 3 
block_Size = 7
#
#blocks= []
#for coords in pts_frame:
#    i = int(coords [0])
#    j = int(coords [1])
#    xcord = i- (block_Size % 2)
#    xend = xcord + block_Size
#    ycord = j - (block_Size % 2)
#    yend =  ycord + block_Size
#
#    block = img2[xcord:xend, ycord:yend] 
#    blocks.append(block)
##    
#plt.imshow(blocks[1], cmap = 'gray')
pts_src = []
pts_dst = []
for i in range(len(good)):
    srcPoint = kp1[ good[i][0].queryIdx ].pt
    dstPoint = kp2[ good[i][0].trainIdx ].pt
    pts_src.append(srcPoint)
    pts_dst.append(dstPoint)

#
#MIN_MATCH_COUNT = 10
#if len(good)>MIN_MATCH_COUNT:
#    src_pts = np.float32(([kp1[idx].pt for idx in range(0, len(kp1))])).reshape(-1,1,2)
#    dst_pts = np.float32(([kp2[idx].pt for idx in range(0, len(kp2))])).reshape(-1,1,2)
#    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#    matchesMask = mask.ravel().tolist()
#    h,w,d = img1.shape
#    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#    dst = cv2.perspectiveTransform(pts,M)
#    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#else:
#    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#    matchesMask = None








