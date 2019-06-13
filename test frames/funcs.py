import numpy as np
import cv2 
from skimage.exposure import cumulative_distribution






def sifter(window, frame):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(window,None)
    kp2, des2 =sift.detectAndCompute(frame,None)
#    print ('des1',des1)
#    print ('des2', des2)
#    if np.any(des1 != None):
#        print ('des1!!!!!!!!!!!!!!!!!!!', des1)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 2)
    search_params = dict(checks = 500)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #bf = cv2.BFMatcher()
    matches = flann.knnMatch(des1,des2,k=2)
    
    
    matches= sorted(matches, key = lambda x:x[0].distance, reverse=True)
    
    good = []
    for m,n in matches:
        if m.distance < 1*n.distance:
            good.append([m])
        
    print ('{} points extracted and reduced to {}'.format(len(matches),len(good)))        
            
    pts_src = []
    pts_dst = []
    for i in range(len(good)):
        srcPoint = kp1[ good[i][0].queryIdx ].pt
        dstPoint = kp2[ good[i][0].trainIdx ].pt
        pts_src.append(srcPoint)
        pts_dst.append(dstPoint)   
 
    
    return pts_src, pts_dst
        
def pt2blk(arr,x_cntr, y_cntr, blkS):
    
    
    i = int(x_cntr)
    j = int(y_cntr)
    
    xstart = int(i- (blkS-1 / 2))
    ystart = int(j - (blkS-1 / 2))
    
    if xstart <0:
        xstart = 0
        
    if ystart < 0: 
        ystart = 0
        
    if xstart+blkS > arr.shape[0]:
        xstart = arr.shape[0]-blkS - 1 
        
    if ystart+blkS >arr.shape[1] :
        ystart = arr.shape[1]-blkS - 1 
    
#    if xstart +blkS > arr.shape[0] or ystart +blkS > arr.shape[1]: 
#        xend = arr.shape[0]-1 
#        yend = arr.shape[1]-1 
#        block = arr[xstart:xend, ystart:yend]
#    else:
#        block = arr[xstart:xend, ystart:yend]
#        
#        block = arr[xcord:xend, ycord:yend]
    block = arr[xstart:xstart+blkS, ystart:ystart+blkS]
    return block

def AbuSMatrix(big_one, small_one, starting_point):
    
    point = list(starting_point)
    blkS = small_one.shape[0]
    
    if point[0]+blkS > big_one.shape[0]:
        point[0] = big_one.shape[0]-blkS
        
    if point[1]+ blkS > big_one.shape[1]:
        point[1] = big_one.shape[1]-blkS
        
    for i in range(big_one.shape[0]):
        for j in range(big_one.shape[1]):
            if i == point[0] and j == point[1]:
                for x in range(small_one.shape[0]):
                    for y in range(small_one.shape[1]):
                        big_one[i + x][j + y] = small_one[x][y]
#                        newimg[i + x][j + y] = newimg[i + x][j + y]+small_one[x][y]
    return big_one




def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def sliding_window2(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
            
def cdf(im):
 '''
 computes the CDF of an image im as 2D numpy ndarray
 '''
 c, b = cumulative_distribution(im) 
 # pad the beginning and ending pixels and their CDF values
 c = np.insert(c, 0, [0]*b[0])
 c = np.append(c, [1]*(255-b[-1]))
 return c

def hist_matching(c, c_t, im):
 '''
 c: CDF of input image computed with the function cdf()
 c_t: CDF of template image computed with the function cdf()
 im: input image as 2D numpy ndarray
 returns the modified pixel values
 ''' 
 pixels = np.arange(256)
 # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
 # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
 new_pixels = np.interp(c, c_t, pixels) 
 im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
 return im

#def sliding_window(image, stepSize, windowSize):
#	# slide a window across the image
#    new_image = []
#    for y in range(0, image.shape[0], stepSize):
#        for x in range(0, image.shape[1], stepSize):
#            # yield the current window
#            print ('processing window {}'.format(y+x))
#            block = image[y:y + windowSize[1], x:x + windowSize[0]]
#            pts_src, pts_dst = sifter(block, image)
#            for src,dst in zip(pts_src,pts_dst):
#                src_blk = get_blocks(image,pts_src[0],pts_src[1])
#                dst_blk = get_blocks(block, pts_dst[0],pts_dst[1])
#                matched = hist_match(dst_blk, src_blk)
#                new_image = AbuSMatrix(image, matched, (dst))
## =============================================================================
##                         histogram match!!! and or filtering here
## =============================================================================
#    return new_image
#



           
