# -*- coding: utf-8 -*-
import numpy as np 
import sys
import cv2 



# Parameters initialization
sigma = 25
Threshold_Hard3D = 1*sigma           
First_Match_threshold = 2500            
Step1_max_matched_cnt = 16             
Step1_Blk_Size = 4                    
Step1_Blk_Step = 3                      
Step1_Search_Step = 3                  
Step1_Search_Window = 39               

Second_Match_threshold = 400           
Step2_max_matched_cnt = 32
Step2_Blk_Size = 8
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39

Beta_Kaiser = 2.0

def load_images(image_paths):

    """
    loads files from proived list (image_paths)
    """

    # load images as grayscale and rescale the images
    sys.stdout.write('Loading images:\n\r')
    sys.stdout.flush()

    images = []
    for path in image_paths:
        sys.stdout.write('Loading image: {} ...'.format(path))
        sys.stdout.flush()
        img =cv2.imread(path, 0)
        images.append(img)
        sys.stdout.write(' Done\n\r')
        sys.stdout.flush()
        
        return images


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


def Step1_fast_match(_noisyImg, _BlockPoint):

    (present_x, present_y) = _BlockPoint 
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = numpy.zeros((max_matched, 2), dtype=int)  
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
    dct_img = cv2.dct(img.astype(np.float64))  

    Final_similar_blocks[0, :, :] = dct_img
    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size-Blk_Size)/Search_Step  
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num**2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num**2, 2), dtype=int)
    Distances = np.zeros(blk_num**2, dtype=float)  

    
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x+Blk_Size, present_y: present_y+Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(np.float64))
            m_Distance = np.linalg.norm((dct_img-dct_Tem_img))**2 / (Blk_Size**2)

            
            if m_Distance < Threshold and m_Distance > 0:  
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

   
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i-1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i-1], :]
    return Final_similar_blocks, blk_positions, Count   