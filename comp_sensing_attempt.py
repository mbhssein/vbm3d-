import arf  
from clairvoyance import img_tools 
import cv2
#from bm3dfunc import BM3D_1st_step,BM3D_2nd_step
import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


 
#import cv2 
filename = r'D:\Frames\frames-TM1490200440GR00-4\frames-TM1490200440GR00-4.arf' 
 
arf_obj = arf.read(filename, 'r') 
 
 
#frame = arf_obj.load(5) 
#scaled_frame = img_tools.scale_frame_by_percentile(frame)#, low_pct=5, high_pct=95) 
#plt.imshow(scaled_frame, cmap  = 'gray')  
 
 

frames = [] 
for n in range(10): 
    frame = arf_obj.load(n) 
    #scaled_frame = img_tools.scale_frame_by_percentile(frame) 
    #list_frames.extend(scaled_frame)
    frame = cv2.resize(frame,None,fx=0.4,fy=0.4)
    frames.append(frame)
    
    
    
    
    


# Load image
original = frames[5]
scales = [1, 2, 3, 4, 10, 15]
# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.cwt(original, scales , 'mexh' )
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()