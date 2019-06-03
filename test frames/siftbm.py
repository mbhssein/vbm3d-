import numpy as np
import cv2
from matplotlib import pyplot as plt
import arf  
from clairvoyance import img_tools 
from funcs import sliding_window2, sifter, pt2blk, hist_match, AbuSMatrix, cdf
import time 
#from skimage.transform import match_histograms
import my_PSNR
#from hist_match import hist_match
#matched = hist_match(source, template)

winS = 256

t1 = time.clock()

file2 = r'D:\Frames\frames-TM1490202529GR00-4\frames-TM1490202529GR00-4.arf'
arf_obj = arf.read(file2, 'r') 
img2 = arf_obj.load(932) 
img2 = img_tools.scale_frame_by_percentile(img2,low_pct=1, high_pct=99)*255
img2 = img2.astype(np.uint8)
blocks = []
for (x, y, window) in sliding_window2(img2, stepSize=64, windowSize=(winS, winS)):
		# if the window does not meet our desired window size, ignore it
    if window.shape[0] != winS or window.shape[1] != winS:
         continue
    print ('mean is', np.mean(window))
    pts_src, pts_dst = sifter(window, img2)
    for src,dst in zip(pts_src,pts_dst):
        src_blk = pt2blk(window,pts_src[0][0],pts_src[0][1])
        dst_blk = pt2blk(img2, pts_dst[0][0],pts_dst[0][1])
        matched = hist_match(dst_blk, src_blk)
#        matched = match_histograms(dst_blk, src_blk, multichannel=False)
#        c = cdf (dst_blk)
#        c_t = cdf (src_blk)
#        matched = hist_matching(c,c_t,dst_blk)
        
        new_image = AbuSMatrix(img2, matched, (dst))
    
    
    
    
    
    
plt.imshow(new_image, cmap = 'gray')
plt.imsave('results/TM1490202529GR00_sift_932.png', new_image, cmap = 'gray')
plt.imsave('results/TM1490202529GR00_orig_932.png', img2     , cmap = 'gray')
#    img1 = window
#    pts_src, pts_dst = sifter(img1, img2)
img_ref= arf_obj.load(1134) 
    
psnr = my_PSNR.PSNR(img_ref, new_image) 
print('PSNR=',psnr)    
    
    
t2 = time.clock

#print ('took {} seconds'.format(t2 - t1))
    