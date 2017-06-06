from __future__ import print_function

import sys
sys.path.append( u'../arrangement/' )
from core import map_alignment as mapali 

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
# prun_dis_threshold = .15 # for home environment - distance image 2
# prun_dis_threshold = .075 # for lab environment - distance image 2

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

scale_mismatch_ratio_threshold = .2 # .5 #.1
scale_bounds = [.5, 2] # [.3,3] [.1, 10]

too_many_tforms = 1000 #30000
dbscan_eps = .051
dbscan_min_samples = 2

for k1_idx in range(4,len(all_keys)):
    for k2_idx in range(k1_idx+1, len(all_keys)):
        keys = [ all_keys[k1_idx], all_keys[k2_idx] ]
    
        if k1_idx < 8:
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
    
        if after_reject >0:
            #################################################### pick the winning hypothesis
            hypothesis, n_cluster = mapali.select_winning_hypothesis(arrangements, keys,
                                                                     tforms,
                                                                     too_many_tforms=too_many_tforms,
                                                                     dbscan_eps=dbscan_eps,
                                                                     dbscan_min_samples=dbscan_min_samples,
                                                                     print_messages=False )
            
            match_score = mapali.arrangement_match_score(arrangements[keys[0]], arrangements[keys[1]], hypothesis)
            
            print ('{:d},{:d} & {:s} & '.format(k1_idx, k2_idx, '_'.join(keys) ) ,
                   '{:d} & {:d} & {:d} & {:.4f} \\\\ '.format(total_t, after_reject, n_cluster, match_score) )
            
            ### visualize the wining transform (ie. hypothesis [or solution?])
            # mapali.maplt.plot_transformed_images(images, keys,
            #                                      tformM= hypothesis.params,
            #                                      title='winning hypothesis',
            #                                      save_to_file=True)
            np.save('_'.join(keys)+'.npy', hypothesis)

        else:
            print ('{:d},{:d} & {:s} & '.format( k1_idx, k2_idx, '_'.join(keys) ) ,
                   '{:d} & {:d} & {:d} & {:.4f}\\\\ '.format(total_t, after_reject, 0, 0) )


