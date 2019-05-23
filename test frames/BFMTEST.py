import numpy as np
import cv2
from matplotlib import pyplot as plt
import arf  
from clairvoyance import img_tools 

file2 = r'D:\Frames\frames-TM1490202529GR00-4\Frames-tm1490202529gr00-4.arf'
 
arf_obj = arf.read(file2, 'r') 
img1 = arf_obj.load(1) 
img1 = img1 [540:592,888:952]

#img1 = img1 [230:323,501:562]


img1 = img_tools.scale_frame_by_percentile(img1, low_pct=1, high_pct=99)*255
img1 = img1.astype(np.uint8)
plt.imshow(img1, cmap = 'gray')
#img2 = img2.astype(np.uint8)
img2 = arf_obj.load(100) 
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


# create BFMatcher object
bf = cv2.BFMatcher()#cv2.NORM_L2, crossCheck=False)

# Match descriptors.
matches = bf.knnMatch(des1,des2, k=2)#bf.match(des1,des2)

# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.92*n.distance:
        good.append([m])
# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None, flags=2)

plt.imshow(img3),plt.show()
plt.imsave('sift_KNN2_096.jpg',img3)
frame_pts = ([kp2[idx].pt for idx in range(0, len(kp2))])
obj_pts = ([kp1[idx].pt for idx in range(0, len(kp1))])