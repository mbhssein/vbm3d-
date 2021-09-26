import matplotlib.pyplot as plt 
#from bm3dfunc import BM3D_1st_step,BM3D_2nd_step
#import utils.PSNR
import cv2
import numpy as np 

 
#import cv2    

gray = #load image
# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
gray[res[:,1],res[:,0]]=[255]
#gray[res[:,3],res[:,2]] = [255]
#
#plt.imshow(gray, cmap = 'gray')

cv2.imshow('window',gray)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
cv2.imwrite('subpixel5.png',img)
