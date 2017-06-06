from __future__ import print_function

import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

new_paths = [
    u'../arrangement/',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import time
import copy
import numpy as np
import scipy
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt

# note: mapali includes plotting methods as mapali.maplt.method()
# note: mapali includes arrangement as mapali.arr (arr contains arr.utls)

from core import map_alignment as mapali 
reload(mapali)
reload(mapali.maplt)

'''
# TODO:

## DEV
[ ] improve optimization performance/speed (menpofit?)
[ ] how would the method perform if place categories are not cosidered (this is the only place that I use place categories)
[ ] improve arrangement_match_score speed

[x] improve f2f_match_score speed (area union/intersection) - used "Polygon" lib
https://pypi.python.org/pypi/Polygon/2.0.4
https://github.com/greginvm/pyclipper

## DOC
[ ] proper docunemtation of the methods in mapali 

## CLEANUP
[ ] move the identified methods to arrangement
[x] identify the methods to move to arrangement
[x] convert the map_alignement to a python package


'''

################################################################################
################################################################################ 
################################################################################

print (4*'\t**************')
#######################################
# mapali.data_sets: is a dictionary storing file names
# mapali.data_sets.keys()

# keys = ['HIH_layout']
keys = [ ['HIH_layout', 'HIH_01_tango'], ['HIH_01_tango'] ][0]
# keys = [ ['HIH_layout', 'HIH_02_tango'], ['HIH_02_tango'] ][0]
# keys = [ ['HIH_layout', 'HIH_03_tango'], ['HIH_03_tango'] ][0]
# keys = [ ['HIH_layout', 'HIH_04_tango'], ['HIH_04_tango'] ][0]

# keys = ['kpt4a_layout']
# keys = [ ['kpt4a_layout', 'kpt4a_f_tango'], ['kpt4a_f_tango'] ] [0]
# keys = [ ['kpt4a_layout', 'kpt4a_kb_tango'], ['kpt4a_kb_tango'] ] [0]
# keys = [ ['kpt4a_layout', 'kpt4a_kl_tango'], ['kpt4a_kl_tango'] ] [0]
# keys = [ ['kpt4a_layout', 'kpt4a_lb_tango'], ['kpt4a_lb_tango'] ] [0]

# keys = ['E5_layout']
# keys = [ ['E5_layout', 'E5_01_tango'], ['E5_01_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_02_tango'], ['E5_02_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_03_tango'], ['E5_03_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_04_tango'], ['E5_04_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_05_tango'], ['E5_05_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_06_tango'], ['E5_06_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_07_tango'], ['E5_07_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_08_tango'], ['E5_08_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_09_tango'], ['E5_09_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_10_tango'], ['E5_10_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_11_tango'], ['E5_11_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_12_tango'], ['E5_12_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_13_tango'], ['E5_13_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_14_tango'], ['E5_14_tango'] ] [0]

# keys = ['F5_layout']
# keys = [ ['F5_layout', 'F5_01_tango'], ['F5_01_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_02_tango'], ['F5_02_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_03_tango'], ['F5_03_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_04_tango'], ['F5_04_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_05_tango'], ['F5_05_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_06_tango'], ['F5_06_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_07_tango'], ['F5_07_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_08_tango'], ['F5_08_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_09_tango'], ['F5_09_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_10_tango'], ['F5_10_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_11_tango'], ['F5_11_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_12_tango'], ['F5_12_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_13_tango'], ['F5_13_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_14_tango'], ['F5_14_tango'] ] [0]

################################################################################
####################################################################### dev yard
################################################################################

######################################## parameters setting
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}

prun_dis_neighborhood = 2
prun_dis_threshold = .15 # for home environment - distance image 2
# prun_dis_threshold = .075 # for lab environment - distance image 2

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

# images, label_images, dis_images, skizs, arrangements, connectivity_maps = mapali.load_and_interpret (keys,
#                                                                                                       arr_config,
#                                                                                                       prun_dis_neighborhood,
#                                                                                                       prun_dis_threshold,
#                                                                                                       con_map_neighborhood,
#                                                                                                       con_map_cross_thr,
#                                                                                                       print_messages=False)

# images, label_images, dis_images, skizs, traits = {}, {}, {}, {}, {}
# arrangements, connectivity_maps = {}, {}
# for key in keys:
#     ######################################## loading file
#     image, label_image, dis_image, skiz, trait = mapali.loader(mapali.data_sets[key])
#     images[key] = image
#     label_images[key] = label_image
#     dis_images[key] = dis_image
#     skizs[key] = skiz
#     traits[key] = trait
#     ######################################## deploying arrangement
#     arrange = mapali.arr.Arrangement(trait, arr_config)
#     ###############  distance based edge pruning
#     mapali.set_edge_distance_value(arrange, dis_image, prun_dis_neighborhood)
#     arrange = mapali.prune_arrangement_with_distance(arrange, dis_image,
#                                                      neighborhood=prun_dis_neighborhood,
#                                                      distance_threshold=prun_dis_threshold)    
#     ######################################## storing results
#     arrangements[key] = arrange
#     # connectivity_maps[key] = con_map

########## plotting
if 1:
    row, col = 1, len(keys)
    fig, axes = plt.subplots(row, col, figsize=(20,12))
    if isinstance(axes, matplotlib.axes.Axes): axes=[axes]

    for idx, key in enumerate(keys):
        ### plotting the ogm, label_image and skiz
        axes[idx].imshow(images[key], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
        # axes[idx].imshow(dis_images[key], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
        # axes[idx].imshow(skizs[key], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
        # axes[idx].imshow(label_images[key], cmap=None, alpha=.7, interpolation='nearest', origin='lower')
        
        ### plotting arrangement and connectivity map
        mapali.maplt.plot_arrangement(axes[idx], arrangements[key], printLabels=False)
        # mapali.maplt.plot_connectivity_map(axes[idx], connectivity_maps[key])

        ### plotting face categories
        # mapali.plot_place_categories(axes[idx], arrangements[key], alpha=.3)

        ### plot edge [occupancy, distance], percent - text
        # mapali.maplt.plot_text_edge_occupancy(axes[idx], arrangements[key],attribute_key=['occupancy'])
        # mapali.maplt.plot_text_edge_occupancy(axes[idx], arrangements[key], attribute_key=['distances'])
        # mapali.maplt.plot_text_edge_occupancy(axes[idx], arrangements[key], attribute_key=['distances', 'occupancy'])

    # fig.savefig('{:s}_{:s}_decomposition'.format(keys[0]))#,keys[1]))
    plt.tight_layout()
    plt.show()

################################################################################
########################################################## Hypothesis generation
################################################################################
print (4*'\t**************'+'_'.join(keys))

tforms = mapali.hypothesis_generator(arrangements, images, keys,
                                     print_messages=True)

################################################################################
#################################################### pick the winning hypothesis
################################################################################
hypothesis = mapali.select_winning_hypothesis(arrangements, keys,
                                              tforms, too_many_tforms=1000,
                                              dbscan_eps=0.051, dbscan_min_samples=2,
                                              print_messages=False )

print ('match_score: {:.4f}'.format(mapali.arrangement_match_score(arrangements[keys[0]], arrangements[keys[1]], hypothesis)) )

### visualize the wining transform (ie. hypothesis [or solution?])
mapali.maplt.plot_transformed_images(images, keys,
                                     tformM= hypothesis.params,
                                     title='winning hypothesis',
                                     save_to_file=True)

