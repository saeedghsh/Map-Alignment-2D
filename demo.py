'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''
from __future__ import print_function

import sys
new_paths = [
    u'../arrangement/',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from map_alignment import map_alignment as mapali
from map_alignment import mapali_plotting as maplt

################################################################################
################################################################################
################################################################################
def _extract_target_file_name(img_src, img_dst, method=None):
    '''
    '''
    spl_src = img_src.split('/')
    spl_dst = img_dst.split('/')
    if len(spl_src)>1 and len(spl_dst)>1:
        # include the current directories name in the target file's name
        tmp = spl_src[-2]+'_'+spl_src[-1][:-4] + '__' + spl_dst[-2]+'_'+spl_dst[-1][:-4]
    else:
        # only include the input files' name in the target file's name
        tmp = spl_src[-1][:-4] + '__' + spl_dst[-1][:-4]

    return tmp if method is None else method+'_'+ tmp


################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    '''    
    list of supported options
    -------------------------
    -visualize
    -save_to_file
    -multiprocessing

    list of supported parameters
    ----------------------------

    NOTE on binarization and occupancy threshold
    --------------------------------------------
    only for trait detection, the theshold is set to 100, so unexplored is
    considered open space. For SKIZ, distance transform, and face occupancy
    ratio, unexplored are considered as occupied and threshold is set to 200.

    example
    -------
    python demo.py --img_src 'tests/maps/map_src.png' --img_dst 'tests/maps/map_dst.png' -multiprocessing -visualize
    python demo.py --img_src 'tests/maps/map_src.png' --img_dst 'tests/maps/map_dst.png' -multiprocessing -save_to_file
    '''    

    args = sys.argv

    ###### fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]

    ###### fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument   
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break   
        
    ##### setting defaults values for visualization and saving options
    visualize = True if 'visualize' in options else False
    save_to_file = True if 'save_to_file' in options else False

    # out_file_name = _extract_target_file_name(img_src, img_dst)
    # save_to_file = out_file_name if save_to_file==True else False
    multiprocessing = True if 'multiprocessing' in options else False

    
    ################################################################################
    print_messages = False
    
    ########## image loading, SKIZ, distance transform and trait detection
    lnl_config = {'binary_threshold_1': 200, # with numpy - for SKIZ and distance
                  'binary_threshold_2': [100, 255], # with cv2 - for trait detection
                  'traits_from_file': False, # else provide yaml file name
                  'trait_detection_source': 'binary_inverted',
                  'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size
                  'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
                  'orthogonal_orientations': True} # for dominant orientation detection


    ########################################
    # img_src = 'tests/maps/map_src.png'
    # img_dst = 'tests/maps/map_dst.png'
    multiprocessing = True
    visualize = True
    ########################################
    
    
    src_results, src_lnl_t = mapali._lock_n_load(img_src, lnl_config)
    dst_results, dst_lnl_t = mapali._lock_n_load(img_dst, lnl_config)

    ########## arrangement (and pruning)
    arr_config = {'multi_processing':4, 'end_point':False, 'timing':False,
                  'prune_dis_neighborhood': 2,
                  'prune_dis_threshold': .075, # home:0.15 - office:0.075
                  'occupancy_threshold': 200} # cell below this is considered occupied
    
    src_results['arrangement'], src_arr_t = mapali._construct_arrangement(src_results, arr_config)
    dst_results['arrangement'], dst_arr_t = mapali._construct_arrangement(dst_results, arr_config)

    # time
    interpret_t = src_lnl_t + src_arr_t + dst_lnl_t + dst_arr_t 
    
    ########## Hypothesis generation
    hyp_config = { 'scale_mismatch_ratio_threshold': .3, # .5,
                   'scale_bounds': [.5, 2], #[.1, 10]
                   'face_occupancy_threshold': .5}
    
    tforms, hyp_gen_t, tforms_total, tforms_after_reject = mapali._generate_hypothese(src_results['arrangement'],
                                                                                      src_results['image'].shape,
                                                                                      dst_results['arrangement'],
                                                                                      dst_results['image'].shape,
                                                                                      hyp_config)

    ########## pick the winning hypothesis
    sel_config = {'multiprocessing': multiprocessing,
                  'too_many_tforms': 3000,
                  'dbscan_eps': 0.051,
                  'dbscan_min_samples': 2}
    hypothesis, n_cluster, sel_win_t = mapali._select_winning_hypothesis(src_results['arrangement'],
                                                                         dst_results['arrangement'],
                                                                         tforms, sel_config)

    details = { 
        'src_lnl_t': src_lnl_t,
        'dst_lnl_t': dst_lnl_t,
        'src_arr_t': src_arr_t,
        'dst_arr_t': dst_arr_t,
        'hyp_gen_t': hyp_gen_t,
        'sel_win_t': sel_win_t,
        'tforms_total': tforms_total,
        'tforms_after_reject': tforms_after_reject,
        'n_cluster': n_cluster
    }

    maplt._visualize_save(src_results, dst_results, hypothesis, visualize, save_to_file, details)
    # maplt._visualize_save_2(src_results, dst_results, hypothesis, visualize, save_to_file, details)

    time_key = ['src_lnl_t', 'dst_lnl_t', 'src_arr_t', 'dst_arr_t', 'hyp_gen_t']
    print ('total time: {:.5f}'.format( np.array([details[key] for key in time_key]).sum() ) )
