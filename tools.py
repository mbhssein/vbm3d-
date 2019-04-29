# -*- coding: utf-8 -*-
import numpy as np 



def getBlocks(img, k):
    """ Break the image up into kxk blocks. Crop if necessary."""

    # Throw an error if not grayscale.
    if len(img.shape) != 2:
        print ("Image is not grayscale. Returning empty block list.")
        return []
    
    blocks = []
    n_vert = img.shape[0] / k
    n_horiz = img.shape[1] / k
    n_vert = int(n_vert)
    n_horiz = int(n_horiz)
    # Iterate through the image and append to 'blocks.'
    for i in range(n_vert):
        for j in range(n_horiz):
            blocks.append(img[i*k:(i+1)*k, j*k:(j+1)*k])

    return blocks


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    
    point_x = _BlockPoint[0]  
    point_y = _BlockPoint[1]  

    #SearchWindow
    LX = point_x+Blk_Size/2-_WindowSize/2     
    LY = point_y+Blk_Size/2-_WindowSize/2     
    RX = LX+_WindowSize                       
    RY = LY+_WindowSize                       

   
    if LX < 0:   LX = 0
    elif RX > _noisyImg.shape[0]:   LX = _noisyImg.shape[0]-_WindowSize
    if LY < 0:   LY = 0
    elif RY > _noisyImg.shape[0]:   LY = _noisyImg.shape[0]-_WindowSize

    return np.array((LX, LY), dtype=int)    