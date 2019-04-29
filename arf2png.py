
import arf 
from clairvoyance import img_tools
from matplotlib import pyplot as plt 
#import cv2
filename = r'D:\Frames\frames-TM1490200440GR00-4\frames-TM1490200440GR00-4.arf'

arf_obj = arf.read(filename, 'r')


#frame = arf_obj.load(5)
#scaled_frame = img_tools.scale_frame_by_percentile(frame)#, low_pct=5, high_pct=95)
#plt.imshow(scaled_frame, cmap  = 'gray') 



list_frames = []
for n in range(902):
    frame = arf_obj.load(n)
    scaled_frame = img_tools.scale_frame_by_percentile(frame)
    plt.imsave('TM1490200440GR00\\frame_{}.png'.format(n),scaled_frame, cmap = 'gray')