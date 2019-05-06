import arf 
import tools
import cv2
import os
from clairvoyance.base import list_folder

# =============================================================================
# Image_path:  the directory to the set of images. use double backslashes.
# number_of_frames: number of images to be used 
# =============================================================================
block_size = 32 
number_of_frames = 5
images_path = 'D:\\Frames\\frames-TM1490200440GR00-4\\TM1490200440GR00'
img_list = list_folder(images_path, extensions = 'png', fullpaths=True)
load_images = img_list[:][:number_of_frames]
# =============================================================================
# 
# =============================================================================

images = [] 
for path in load_images: 
    img =cv2.imread(path,0) 
    images.append(img) 
# =============================================================================
# images: a list of all input images.     
# =============================================================================
 # create block segments from the focal stack images
block_shape = (block_size, block_size)
block_list = []
for img in images:
    blocks =tools.getBlocks(img, block_shape[0])
    block_list.extend(blocks)


num_blocks = len (block_list) 
#here is the search for similatrity step...... 
#look at BM3d algorithms, notes, querey function from pandas , and lambda from scikit
for block in block_list:    
 ###do filtering
####do threshold 
####maybe clahe
#some similarity measure
dct_img = cv2.dct(block.astype(numpy.float64)) 
