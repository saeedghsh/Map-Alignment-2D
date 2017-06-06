from __future__ import print_function

import sys

import cv2
import numpy as np
import matplotlib.transforms
import matplotlib.pyplot as plt

################################################################################
def _register_image_ORB(src, dst,        ):
    '''
    '''
    return None

################################################################################
def _visualize_save_ORB(src, dst, warp_matrix, warp_type, visualize, save_to_file):
    '''
    if 'save_to_file' is not False, it must be a string containing the file name
    to save to (for both figure and warp_matrix).

    '''
    pass

################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    '''
    python align_images.py --img_src 'filename1.ext' --img_dst 'filename2.ext' -save_to_file

    # list of supported options
    -visualize
    -save_to_file

    # list of supported parameters

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

    ### load images
    src = cv2.imread(img_src, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(img_dst, cv2.IMREAD_GRAYSCALE)
    
    

