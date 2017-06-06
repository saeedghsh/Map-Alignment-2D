from __future__ import print_function

import sys
sys.path.append( u'../arrangement/' )
from core import map_alignment as mapali 
import numpy as np
################################################################################
################################################################################
################################################################################
key_pairs = { 1: ['HIH_layout', 'HIH_01_tango'], 
              2: ['HIH_layout', 'HIH_02_tango'],
              3: ['HIH_layout', 'HIH_03_tango'],
              4: ['HIH_layout', 'HIH_04_tango'],
              5: ['kpt4a_layout', 'kpt4a_f_tango'],
              6: ['kpt4a_layout', 'kpt4a_kb_tango'],
              7: ['kpt4a_layout', 'kpt4a_kl_tango'],
              8: ['kpt4a_layout', 'kpt4a_lb_tango'],
              
              9: ['E5_layout', 'E5_01_tango'],
              10: ['E5_layout', 'E5_02_tango'],
              11: ['E5_layout', 'E5_03_tango'],
              12: ['E5_layout', 'E5_04_tango'],
              13: ['E5_layout', 'E5_05_tango'],
              14: ['E5_layout', 'E5_06_tango'],
              15: ['E5_layout', 'E5_07_tango'],
              16: ['E5_layout', 'E5_08_tango'],
              17: ['E5_layout', 'E5_09_tango'],
              18: ['E5_layout', 'E5_10_tango'],
              19: ['E5_layout', 'E5_11_tango'],
              20: ['E5_layout', 'E5_12_tango'],
              21: ['E5_layout', 'E5_13_tango'],
              22: ['E5_layout', 'E5_14_tango'],
              23: ['F5_layout', 'F5_01_tango'],
              24: ['F5_layout', 'F5_02_tango'],
              25: ['F5_layout', 'F5_03_tango'],
              26: ['F5_layout', 'F5_04_tango'],
              27: ['F5_layout', 'F5_05_tango'],
              28: ['F5_layout', 'F5_06_tango'],
              29: ['F5_layout', 'F5_07_tango'],
              30: ['F5_layout', 'F5_08_tango'],
              31: ['F5_layout', 'F5_09_tango'],
              32: ['F5_layout', 'F5_10_tango'],
              33: ['F5_layout', 'F5_11_tango'],
              34: ['F5_layout', 'F5_12_tango'],
              35: ['F5_layout', 'F5_13_tango'],
              36: ['F5_layout', 'F5_14_tango'] }

######################################## parameters setting
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}

prun_dis_neighborhood = 2
# prun_dis_threshold = .15 # for home environment - distance image 2
# prun_dis_threshold = .075 # for lab environment - distance image 2

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

scale_mismatch_ratio_threshold = 0.2 #.1 # 0.5 # .1
scale_bounds = [.5, 2] # [.3,3] # [.1, 10] #

too_many_tforms = 30000 # 1000 #
dbscan_eps = .051
dbscan_min_samples = 2

for idx in key_pairs: #[24,32]
    keys = key_pairs[idx]
    
    if idx <9:
        prun_dis_threshold = .15 # for home environment - distance image 2
    else:
        prun_dis_threshold = .075 # for lab environment - distance image 2
        
    images, _, _, _, arrangements, connectivity_maps = mapali.load_and_interpret (keys,
                                                                                  arr_config,
                                                                                  prun_dis_neighborhood,
                                                                                  prun_dis_threshold,
                                                                                  con_map_neighborhood,
                                                                                  con_map_cross_thr,
                                                                                  print_messages=False)

    ########################################################## Hypothesis generation
    tforms, (total_t, after_reject) = mapali.hypothesis_generator(arrangements, images, keys,
                                                                  scale_mismatch_ratio_threshold=scale_mismatch_ratio_threshold,
                                                                  scale_bounds=scale_bounds,
                                                                  print_messages=False)

    
    
    #################################################### pick the winning hypothesis
    hypothesis, n_cluster = mapali.select_winning_hypothesis(arrangements, keys,
                                                             tforms,
                                                             too_many_tforms=too_many_tforms,
                                                             dbscan_eps=dbscan_eps,
                                                             dbscan_min_samples=dbscan_min_samples,
                                                             print_messages=False )

    match_score = mapali.arrangement_match_score(arrangements[keys[0]], arrangements[keys[1]], hypothesis)

    print (keys[1], '&',
           '{:d}'.format(total_t) ,'&',
           '{:d}'.format(after_reject) ,'&',
           '{:d}'.format(n_cluster) ,'&',
           '{:.4f}'.format(match_score), '\\')


    # ### visualize the wining transform (ie. hypothesis [or solution?])
    # mapali.maplt.plot_transformed_images(images, keys,
    #                                      tformM= hypothesis.params,
    #                                      title='winning hypothesis',
    #                                      save_to_file=True)
    # np.save('_'.join(keys)+'.npy', hypothesis)
