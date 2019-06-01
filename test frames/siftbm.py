import numpy as np
import cv2
from matplotlib import pyplot as plt
import arf  
from clairvoyance import img_tools 
from funcs import sliding_window2, sifter 
import time 
#from hist_match import hist_match
#matched = hist_match(source, template)

winS = 128

t1 = time.clock()

file2 = r'D:\Frames\frames-TM1490202529GR00-4\Frames-tm1490202529gr00-4.arf'
arf_obj = arf.read(file2, 'r') 
img2 = arf_obj.load(100) 
img2 = img_tools.scale_frame_by_percentile(img2,low_pct=1, high_pct=99)*255
img2 = img2.astype(np.uint8)
blocks = []
for (x, y, window) in sliding_window2(img2, stepSize=32, windowSize=(winS, winS)):
		# if the window does not meet our desired window size, ignore it
    if window.shape[0] != winS or window.shape[1] != winS:
         continue
    
    plt.imshow(window)
#    img1 = window
#    pts_src, pts_dst = sifter(img1, img2)

    
    
    
    
t2 = time.clock

#print ('took {} seconds'.format(t2 - t1))
    