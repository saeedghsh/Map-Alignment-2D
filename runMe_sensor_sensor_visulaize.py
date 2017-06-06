from __future__ import print_function

import sys
sys.path.append( u'../arrangement/' )
from core import map_alignment as mapali 

import cv2
import numpy as np

################################################################################
################################################################################ 
################################################################################
all_keys = ['HIH_01_tango', 'HIH_02_tango', 'HIH_03_tango', 'HIH_04_tango',
            'kpt4a_f_tango', 'kpt4a_kb_tango', 'kpt4a_kl_tango', 'kpt4a_lb_tango',
            'E5_01_tango', 'E5_02_tango', 'E5_03_tango', 'E5_04_tango', 'E5_05_tango',
            'E5_06_tango', 'E5_07_tango', 'E5_08_tango', 'E5_09_tango', 'E5_10_tango',
            'E5_11_tango', 'E5_12_tango', 'E5_13_tango', 'E5_14_tango', 
            'F5_01_tango', 'F5_02_tango', 'F5_03_tango', 'F5_04_tango', 'F5_05_tango',
            'F5_06_tango', 'F5_07_tango', 'F5_08_tango', 'F5_09_tango', 'F5_10_tango',
            'F5_11_tango', 'F5_12_tango', 'F5_13_tango', 'F5_14_tango']

################################################################################
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}

prun_dis_neighborhood = 2
prun_dis_threshold = .15 # for home environment - distance image 2
# prun_dis_threshold = .075 # for lab environment - distance image 2

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

scale_mismatch_ratio_threshold = .2 # .5 #.1
scale_bounds = [.5, 2] # [.3,3] [.1, 10]

too_many_tforms = 1000 #30000
dbscan_eps = .051
dbscan_min_samples = 2

for k1_idx in range(2,len(all_keys)):
    for k2_idx in range(k1_idx+1, len(all_keys)):

        keys = [ all_keys[k1_idx], all_keys[k2_idx] ]    
        print ( 'loading  {:s}'.format('_'.join(keys)) )

        images = {}
        for key in keys:
            image = np.flipud( cv2.imread( mapali.data_sets[key], cv2.IMREAD_GRAYSCALE) )
            thr1,thr2 = [200, 255]
            ret, image = cv2.threshold(image.astype(np.uint8) , thr1,thr2 , cv2.THRESH_BINARY)
            images[key] = image

        try:
            hypothesis = np.atleast_1d( np.load( '_'.join(keys)+'.npy' ) )[0]
            mapali.maplt.plot_transformed_images(images, keys,
                                                 tformM= hypothesis.params,
                                                 title='winning hypothesis',
                                                 save_to_file=True)
        except:
            print ( 'no hypothesis found for {:s}'.format('_'.join(keys)) )
        
        
