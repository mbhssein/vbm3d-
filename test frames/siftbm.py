import numpy as np
#import cv2
from matplotlib import pyplot as plt
import arf  
from clairvoyance import img_tools 
from funcs import sliding_window2, sifter, pt2blk,  AbuSMatrix, hist_match #,hist_matching , cdf
import time 
#from skimage.transform import match_histograms
import my_PSNR
#from hist_match import hist_match
#matched = hist_match(source, template)



t1 = time.clock()

# =============================================================================
# import and scale image 
file2 = r'D:\Frames\frames-tm1490200440gr00-4\Frames-tm1490200440gr00-4.arf' 
arf_obj = arf.read(file2, 'r') 
img2 = arf_obj.load(200) 

img2 = img_tools.scale_frame_by_percentile(img2,low_pct=1, high_pct=99)*255
img2 = img2.astype(np.uint8)

ref_img= arf_obj.load(887) 
ref_img = img_tools.scale_frame_by_percentile(ref_img,low_pct=1, high_pct=99)*255
ref_img = ref_img.astype(np.uint8)
# =============================================================================
winS = 50
blocks = []
# =============================================================================
# Matching first step
new_img = img2.copy()

for (x, y, window) in sliding_window2(new_img, stepSize=64, windowSize=(winS, winS)):
		# if the window does not meet our desired window size, ignore it
    if window.shape[0] != winS or window.shape[1] != winS:
         continue
    print ('window mean is', np.mean(window))        
    pts_src, pts_dst = sifter(window, new_img)
    for src,dst in zip(pts_src,pts_dst):
        print (int(dst[0]), int(dst[1]))
        src_blk = pt2blk(window,pts_src[0][0],pts_src[0][1])
        dst_blk = pt2blk(ref_img, pts_dst[0][0],pts_dst[0][1])
        matched = hist_match(src_blk, dst_blk)
        plt.imsave('blocks/block{}.jpg'.format(dst[0]),matched, cmap = 'gray')
        matched = np.zeros([500,500])
#        matched = match_histograms(dst_blk, src_blk, multichannel=False)
#        c = cdf (dst_blk)
#        c_t = cdf (src_blk)
#        matched = hist_matching(c,c_t,dst_blk)
        new_img = AbuSMatrix(new_img, matched, (dst))
        
        
# =============================================================================    
 
## =============================================================================
## Second matching step
#for (x, y, window) in sliding_window2(new_image, stepSize=32, windowSize=(winS, winS)):
#		# if the window does not meet our desired window size, ignore it
#    if window.shape[0] != winS or window.shape[1] != winS:
#         continue
#    print ('mean is', np.mean(window))    
##    if np.mean(window)<70:
##        
#    pts_src, pts_dst = sifter(window, img2)
#    for src,dst in zip(pts_src,pts_dst):
#        src_blk = pt2blk(window,pts_src[0][0],pts_src[0][1])
#        dst_blk = pt2blk(img2, pts_dst[0][0],pts_dst[0][1])
#        matched = hist_match(dst_blk, src_blk)
##        matched = np.zeros([32,32])
##        matched = match_histograms(dst_blk, src_blk, multichannel=False)
#        c = cdf (dst_blk)
#        c_t = cdf (src_blk)
#        matched = hist_matching(c,c_t,dst_blk)
#        
#        new_2image = AbuSMatrix(img2, matched, (dst))
#        
    
    
    
    
plt.imshow(new_img, cmap = 'gray')
fig=np.concatenate((img2,new_img),axis=0)

# Save without a cmap, to preserve the ones you saved earlier
plt.imsave('figc.png',fig ,cmap= 'gray')
plt.imsave('results/TM1490202529GR00_sift_932.png', new_img   , cmap = 'gray')
#plt.imsave('results/TM1490202529GR00_sift_932.png', new_2image, cmap = 'gray')
plt.imsave('results/TM1490202529GR00_orig_932.png', fig       , cmap = 'gray')
#    img1 = window
#    pts_src, pts_dst = sifter(img1, img2)

    
psnr = my_PSNR.PSNR(ref_img, new_img) 
print('PSNR=',psnr)    
    
    
t2 = time.clock()

print ('took {} seconds'.format(t2 - t1))
    