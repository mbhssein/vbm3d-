import arf 
import tools
import cv2

filename = r'D:\Frames\frames-TM1490200440GR00-4\frames-TM1490200440GR00-4.arf'
arf_obj = arf.read(filename, 'r')

 
frames_to_be_processed = 5
block_size = 32 



list_frames = []
for n in range(frames_to_be_processed):
    frame = arf_obj.load(n)
    list_frames.append(frame)

cv2.imshow('frame',list_frames[1])
#for instant in range(frames_to_be_processed):
    
     # create block segments from the focal stack images
    block_shape = (block_size, block_size)
    block_list = []
    for img in list_frames:
        blocks =tools.getBlocks(img, block_shape[0])
        block_list.extend(blocks)


  
num_blocks = len (block_list) 
#here is the search for similatrity step...... look at BM3d algorithms, notes, querey function from pandas , and lambda from scikit
for block in block_list:    
    
    