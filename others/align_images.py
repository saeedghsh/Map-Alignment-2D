from __future__ import print_function

import sys
import time
import cv2
import numpy as np
import matplotlib.transforms
import matplotlib.pyplot as plt

################################################################################
def _align_image_LK(src, dst):
    # http://docs.opencv.org/trunk/d7/d8b/tutorial_py_lucas_kanade.html
    pass

################################################################################
def _align_image_ECC(src, dst, 
                     warp_type,
                     number_of_iterations,
                     termination_eps
             ):
    '''
    http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    Image Registration using Enhanced Correlation Coefficient (ECC) Maximization
    
    http://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#findtransformecc
    cv2.findTransformECC(templateImage, inputImage, warpMatrix[, motionType[, criteria]]) -> retval, warpMatrix
    templateImage: dst
    inputImage: src (to be warped)
    '''

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_type == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
 
    termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    tic = time.time()
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (dst, src, warp_matrix, warp_type, termination_criteria)
    elapsed_time = time.time() - tic

    return cc, warp_matrix, elapsed_time

################################################################################
def _visualize_save_ECC(src, dst, warp_matrix, warp_type, visualize, save_to_file):
    '''
    if 'save_to_file' is not False, it must be a string containing the file name
    to save to (for both figure and warp_matrix).

    '''
    ### warping source image with opencv
    warp_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    shp = dst.shape
    if warp_type == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        src_aligned = cv2.warpPerspective (src, warp_matrix, (shp[1],shp[0]), flags=warp_flags)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        src_aligned = cv2.warpAffine(src, warp_matrix, (shp[1],shp[0]), flags=warp_flags)

    # plotting
    fig, axes = plt.subplots(1, 1, figsize=(12,12))
    axes.imshow(dst, cmap='gray', alpha=.5)
    axes.imshow(src_aligned, cmap='gray', alpha=.5)

    if visualize:
        plt.tight_layout()
        plt.show()

    if save_to_file:
        np.save(save_to_file+'.npy', warp_matrix)
        plt.savefig(save_to_file+'.png', bbox_inches='tight')
        plt.close(fig)

################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    '''
    python align_images.py --img_src 'filename1.ext' --img_dst 'filename2.ext' --method ECC -save_to_file

    # list of supported options
    -visualize
    -save_to_file

    # list of supported parameters
    --warp_type (default: 1)
    0: cv2.MOTION_TRANSLATION, # two parameters - x, y.
    1: cv2.MOTION_EUCLIDEAN, # three parameters - x, y and angle.
    2: cv2.MOTION_AFFINE, # a combination of rotation, translation ( shift ), scale, and shear.
    3: cv2.MOTION_HOMOGRAPHY # A homography transform can account for some 3D effects.

    --number_of_iterations (default: 5000)
    Specify the number of iterations.
        
    --termination_eps (default: 1e-10)
    Specify the threshold of the increment in the correlation coefficient between two iterations
    '''
    
    args = sys.argv

    # fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]

    # fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument   
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = listiterator.next()
            if item[:2] == '--':
                exec(item[2:] + ' = listiterator.next()')
        except:
            break   

    ### setting defaults values for undefined parameters 
    if 'method' not in locals().keys():
        method = 'ECC'
    if 'warp_type' not in locals().keys():
        warp_type = 1 # cv2.MOTION_EUCLIDEAN (three parameters - x, y and angle.)
    if 'number_of_iterations' not in locals().keys():
        number_of_iterations = 5000
    if 'termination_eps' not in locals().keys():
        termination_eps = 1e-10

    ### setting defaults values for visualization and saving options
    visualize = True if 'visualize' in options else False
    save_to_file = True if 'save_to_file' in options else False
    if save_to_file:
        spl_src = img_src.split('/')
        spl_dst = img_dst.split('/')
        save_to_file = spl_src[-2]+'_'+spl_src[-1][:-4] + '__' + spl_dst[-2]+'_'+spl_dst[-1][:-4]

    ### load images
    src = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(img_dst, cv2.IMREAD_GRAYSCALE)
    
    ### alignment
    if method == 'ECC':
        cc, warp_matrix = _align_image_ECC(src,dst, warp_type,number_of_iterations,termination_eps)
        _visualize_save_ECC(src, dst, warp_matrix, warp_type, visualize, save_to_file)
    elif method == 'LK':
        raise(NameError('LK not implemented yet'))
    else:
        raise(NameError('unsupported method'))

    

