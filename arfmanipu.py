 
import arf  
from clairvoyance import img_tools 
import matplotlib.pyplot as plt 
from utils.bm3dfunc import BM3D_1st_step,BM3D_2nd_step
from utils import PSNR
import cv2

 
#import cv2 
#filename = r'D:\Frames\frames-TM1490200440GR00-4\frames-TM1490200440GR00-4.arf' 
file2 = r'D:\Frames\frames-TM1490202529GR00-4\Frames-tm1490202529gr00-4.arf'
 
arf_obj = arf.read(file2, 'r') 
 
 
#frame = arf_obj.load(5) 
#scaled_frame = img_tools.scale_frame_by_percentile(frame)#, low_pct=5, high_pct=95) 
#plt.imshow(scaled_frame, cmap  = 'gray')  
 
 

frames = [] 
for n in range(5): 
    frame = arf_obj.load(n) 
    #frame = img_tools.scale_frame_by_percentile(frame) 
    #list_frames.extend(scaled_frame)
    #frame = cv2.resize(frame,None,fx=0.8,fy=0.8)
    frames.append(frame)
 
# =============================================================================
#cv2.imshow('nor_noisy_frame.jpg',this_frame) 
#cv2.waitKey()
# =============================================================================

cv2.setUseOptimized(True)  
e1 = cv2.getTickCount()   
Basic_img = BM3D_1st_step(frames [4], frames) 
scaled_b=img_tools.scale_frame_by_percentile(Basic_img) 


e2 = cv2.getTickCount() 
time = (e2 - e1) / cv2.getTickFrequency()   
print ("The Processing time of the First step is %f s" % time) 



psnr = PSNR.PSNR(frames[0], Basic_img) 
print ("The PSNR between the two img of the First step is %f" % psnr) 
 

 
# =============================================================================
#cv2.imshow("Basic31.jpg", scaled_b) 
#cv2.waitKey()

plt.imshow(scaled_b, cmap = 'gray') 
plt.imsave('basic_DCT2_{}.jpg'.format(psnr),scaled_b, cmap = 'gray') 
# =============================================================================


Final_img = BM3D_2nd_step(Basic_img, frames[0]) 
e3 = cv2.getTickCount() 
time = (e3 - e2) / cv2.getTickFrequency() 
print ("The Processing time of the Second step is %f s" % time) 


scaled_f=img_tools.scale_frame_by_percentile(Final_img) 

psnr = PSNR.PSNR(frames[4], Final_img) 
print ("The PSNR between the two img of the Second step is %f" % psnr) 
time = (e3 - e1) / cv2.getTickFrequency()    
print ("The total Processing time is %f s" % time) 

plt.imshow(scaled_f, cmap = 'gray') 

plt.imsave('final_DCT2_{}.jpg'.format(psnr), scaled_f, cmap = 'gray')
 


##cleared_img = BM3Dfunc (list_frames[3],list_frames)

