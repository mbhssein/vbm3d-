 
import arf  
from clairvoyance import img_tools 
import cv2
from bm3dfunc import BM3Dfunc

#import cv2 
filename = r'D:\Frames\frames-TM1490200440GR00-4\frames-TM1490200440GR00-4.arf' 
 
arf_obj = arf.read(filename, 'r') 
 
 
#frame = arf_obj.load(5) 
#scaled_frame = img_tools.scale_frame_by_percentile(frame)#, low_pct=5, high_pct=95) 
#plt.imshow(scaled_frame, cmap  = 'gray')  
 
 
 
list_frames = [] 
for n in range(4): 
    frame = arf_obj.load(n) 
    scaled_frame = img_tools.scale_frame_by_percentile(frame) 
    list_frames.append(scaled_frame)
    
    
#cv2.imshow('frame',list_frames[1]) 
#cv2.waitKey()

cleared_img = BM3Dfunc (list_frames[3],list_frames)

