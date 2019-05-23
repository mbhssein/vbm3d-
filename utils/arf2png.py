

import arf 
from clairvoyance import img_tools
#from matplotlib import pyplot as plt 
#import cv2
filename = r'D:\Frames\frames-TM1490200440GR00-4\frames-TM1490200440GR00-4.arf'
folder = 'D:\Frames\frames-TM1490200440GR00-4'
arf_obj = arf.read(filename, 'r')


#frame = arf_obj.load(5)
#scaled_frame = img_tools.scale_frame_by_percentile(frame)#, low_pct=5, high_pct=95)
#plt.imshow(scaled_frame, cmap  = 'gray') 
n=100


images = []
for n in range(100):
    frame = arf_obj.load(n)
    scaled_frame = img_tools.scale_frame_by_percentile(frame)
    images.append(scaled_frame)
    
    
    
img_tools.write_images_to_dir(images, folder, filetype='.jpg', nametype='sequential')