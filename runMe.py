from __future__ import print_function

import os
import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

new_paths = [
    u'../arrangement/',
    # u'../Python-CPD/',
    # u'../place_categorization_2D',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

# sys.path.append( u'/usr/share/inkscape/extensions' )
# import inkex

import time
import copy
# import itertools
# import operator
# import cv2
import numpy as np
# import sympy as sym
import scipy
# import networkx as nx
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt

# import arrangement.arrangement as arr # ---> import this inside mapali and use it as mapali.arr.[]
import map_alignment as mapali
reload(mapali)

################################################################################
################################################################################ 
################################################################################

print (4*'\t**************')
#######################################
# mapali.data_sets: is a dictionary storing file names
# mapali.data_sets.keys()
# keys = ['HIH_layout']
keys = [ ['HIH_layout', 'HIH_tango'], ['HIH_tango'] ][0]

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

# keys = ['F5_layout']
# keys = [ ['F5_layout', 'F5_01_tango'], ['F5_01_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_02_tango'], ['F5_02_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_03_tango'], ['F5_03_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_04_tango'], ['F5_04_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_05_tango'], ['F5_05_tango'] ] [0]
# keys = [ ['F5_layout', 'F5_06_tango'], ['F5_06_tango'] ] [0]

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

######################################## deployment
images, label_images, dis_images, skizs, traits = {}, {}, {}, {}, {}
arrangements, connectivity_maps = {}, {}


for key in keys:
    print ('\t *** processing map \'{:s}\':'.format(key))

    ######################################## loading file
    print ('\t loading files [image, label_image, skiz, triat_data] ...')
    image, label_image, dis_image, skiz, trait = mapali.loader(mapali.data_sets[key])
    images[key] = image
    label_images[key] = label_image
    dis_images[key] = dis_image
    skizs[key] = skiz
    traits[key] = trait

    ######################################## deploying arrangement
    print ('\t deploying arrangement ... ')
    arrange = mapali.arr.Arrangement(trait, arr_config)

    ############### testing distance based edge pruning
    print ('\t arrangement pruning ... ')
    mapali.set_edge_distance_value(arrange, dis_image, prun_dis_neighborhood)
    arrange = mapali.prune_arrangement_with_distance(arrange, dis_image,
                                                     neighborhood=prun_dis_neighborhood,
                                                     distance_threshold=prun_dis_threshold)

    ######################################## updating faces label
    # due to the changes of the arrangement and faces
    print ('\t update place categories to faces assignment ...')
    arrange = mapali.assign_label_to_all_faces(arrange, label_image)

    ######################################## 
    print ('\t setting ombb attribute of faces ...') 
    arrange = mapali.set_ombb_of_faces (arrange)

    ######################################## construct conectivity map
    print ('\t connectivity map construction and node profiling ...')
    mapali.set_edge_crossing_attribute(arrange, skiz,
                                       neighborhood=con_map_neighborhood,
                                       cross_thr=con_map_cross_thr)
    arrange, con_map = mapali.construct_connectivity_map(arrange, set_coordinates=True)

    # profiling node, for finding label association with other maps
    con_map = mapali.profile_nodes(con_map)

    ######################################## storing results
    arrangements[key] = arrange
    connectivity_maps[key] = con_map

########## plotting
if 0:
    row, col = 1, len(keys)
    fig, axes = plt.subplots(row, col, figsize=(20,12))
    if isinstance(axes, matplotlib.axes.Axes): axes=[axes]

    for idx, key in enumerate(keys):
        ### plotting the ogm, label_image and skiz
        axes[idx].imshow(label_images[key], cmap=None, alpha=.7, interpolation='nearest', origin='lower')
        axes[idx].imshow(images[key], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
        axes[idx].imshow(dis_images[key], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
        axes[idx].imshow(skizs[key], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
        
        ### plotting arrangement and connectivity map
        mapali.maplt.plot_arrangement(axes[idx], arrangements[key], printLabels=False)
        # mapali.maplt.plot_connectivity_map(axes[idx], connectivity_maps[key])

        ### plotting face categories
        # mapali.plot_place_categories(axes[idx], arrangements[key], alpha=.3)

        ### plot edge [occupancy, distance], percent - text
        # mapali.maplt.plot_text_edge_occupancy(axes[idx], arrangements[key],attribute_key=['occupancy'])
        # mapali.maplt.plot_text_edge_occupancy(axes[idx], arrangements[key], attribute_key=['distances'])
        # mapali.maplt.plot_text_edge_occupancy(axes[idx], arrangements[key], attribute_key=['distances', 'occupancy'])

    # fig.savefig('{:s}_{:s}_decomposition'.format(keys[0],keys[1]))
    plt.tight_layout()
    plt.show()

################################################################################
########################################################## Hypothesis generation
################################################################################
print (4*'\t**************')

label_associations = mapali.label_association(arrangements, connectivity_maps)

tforms = []
for face_src in arrangements[keys[0]].decomposition.faces:
    for face_dst in arrangements[keys[1]].decomposition.faces:
        src_not_outlier = face_src.attributes['label_vote'] != -1
        dst_not_outlier = face_dst.attributes['label_vote'] != -1
        similar_labels = True
        similar_labels = label_associations[ face_src.attributes['label_vote'] ] == face_dst.attributes['label_vote']
        if src_not_outlier and src_not_outlier and similar_labels:
            tforms.extend (mapali.align_ombb(face_src,face_dst, tform_type='affine'))

tforms = np.array(tforms)
print ( '\t totaly {:d} transformations estimated'.format(tforms.shape[0]) )
 
tforms = mapali.reject_implausible_transformations( tforms,
                                                    images[keys[0]].shape, images[keys[1]].shape,
                                                    scale_mismatch_ratio_threshold=.5,
                                                    scale_bounds=[.3, 3] )#[.1, 10] )

print ( '\t and {:d} transformations survived the rejections...'.format(tforms.shape[0]) )
if tforms.shape[0] == 0: raise (NameError('no transformation survived.... '))


################################################################################
#################################################### pick the winning hypothesis
################################################################################
if tforms.shape[0] < 100:
    #### if tforms.shape[0] <50: no need to cluster
    print ('only {:d} tforms are estimated, so no clustering'.format(tforms.shape[0]))
        
    arr_match_score = {}
    for idx, tf in enumerate(tforms):
        arrange_src = copy.deepcopy(arrangements[keys[0]])
        arrange_dst = copy.deepcopy(arrangements[keys[1]])
        arr_match_score[idx] = mapali.arrangement_match_score_fast(arrange_src,
                                                                   arrange_dst,
                                                                   tf)# ,
                                                                   # label_associations)
        print ('computing match_score for transform-{:d}/{:d}: {:.4f}'.format(idx,tforms.shape[0], arr_match_score[idx]))

    best_idx = max(arr_match_score, key=arr_match_score.get)
    hypothesis = tforms[best_idx]

else:
    ################################ clustering transformations and winner selection

    #################### feature scaling (standardization)
    parameters = np.stack([ np.array( [tf.translation[0], tf.translation[1], tf.rotation, tf.scale[0] ] )
                            for tf in tforms ], axis=0)
    assert not np.any( np.isnan(parameters))

    # note that the scaling is only applied to parametes (for clustering),
    # the transformations themselves are intact
    parameters -= np.mean( parameters, axis=0 )
    parameters /= np.std( parameters, axis=0 )

    #################### clustering pool into hypotheses
    import sklearn.cluster
    print ('\t clustering {:d} transformations'.format(parameters.shape[0]))
    # min_s = max ([ 2, int( .05* np.min([ len(arrangements[key].decomposition.faces) for key in keys ]) ) ])
    min_s = 2
    cls = sklearn.cluster.DBSCAN(eps=0.051, min_samples=min_s)
    cls.fit(parameters)
    labels = cls.labels_
    unique_labels = np.unique(labels)
    print ( '\t *** total: {:d} clusters...'.format(unique_labels.shape[0]-1) )

    ###  match_score for each cluster
    arr_match_score = {}
    for lbl in np.setdiff1d(unique_labels,[-1]):
        class_member_idx = np.nonzero(labels == lbl)[0]
        class_member = [ tforms[idx] for idx in class_member_idx ]

        # pick the one that is closest to all others in the same group
        params = parameters[class_member_idx]
        dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(params, 'euclidean'))
        dist_arr = dist_mat.sum(axis=0)
        tf = class_member[ np.argmin(dist_arr) ]

        arrange_src = copy.deepcopy(arrangements[keys[0]])
        arrange_dst = copy.deepcopy(arrangements[keys[1]])
        arr_match_score[lbl] = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf)#, label_associations)
        print ('cluster {:d}/{:d} arrangement match score: {:.4f}'.format(lbl,len(unique_labels)-1, arr_match_score[lbl]) )

    ### pick the winning cluster
    winning_cluster_idx = max(arr_match_score, key=arr_match_score.get)
    winning_cluster = [tforms[idx] for idx in np.nonzero(labels==winning_cluster_idx)[0]]

    ### match_score for the entities of the winning cluster
    arr_match_score = {}
    for idx, tf in enumerate(winning_cluster):
        arrange_src = copy.deepcopy(arrangements[keys[0]])
        arrange_dst = copy.deepcopy(arrangements[keys[1]])
        arr_match_score[idx] = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf)#, label_associations)
        print ('element {:d}/{:d} arrangement match score: {:.4f}'.format(idx,len(winning_cluster)-1, arr_match_score[idx]) )

    ### pick the wining cluster
    hypothesis_idx = max(arr_match_score, key=arr_match_score.get)
    hypothesis =  winning_cluster[hypothesis_idx]


### visualize the wining transform (ie. hypothesis [or solution?])
mapali.maplt.plot_transformed_images(images[keys[0]], images[keys[1]],
                                     tformM= hypothesis.params,
                                     title='winning transform')


# ########################################
# ######## optimizing with distance images
# ########################################
# tform = hypothesis

# src_img = dis_images[keys[0]]
# dst_img = dis_images[keys[1]]

# X0 = (tform.translation[0], tform.translation[1], tform.scale[0], tform.rotation)
# # X_bounds = ((None,None),(None,None),(None,None),(None,None)) # No bounds
# # X_bounds = ((X0[0]-100,X0[0]+100),(X0[1]-100,X0[1]+100),
# #             (X0[2]-.1, X0[2]+.1), (X0[3]-.08,X0[3]+.08))
# methods = [ 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
#             'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']#[1,5,6,7,8,]
# # [4,9,10]: need jac
# # [0,2,3]: did not converge

# result = scipy.optimize.minimize( mapali.objectivefun_image, X0,
#                                   args=(src_img, dst_img),
#                                   method = methods[1],
#                                   # bounds = X_bounds,
#                                   tol=1e-6,
#                                   options={'maxiter':100, 'disp':True} )

# if result['success']:
#     fig, axes = plt.subplots(1,2, figsize=(20,12))
    
#     arrange_src = copy.deepcopy(arrangements[keys[0]])
#     arrange_dst = copy.deepcopy(arrangements[keys[1]])
#     match_score_ini = mapali.arrangement_match_score_fast(arrange_src,
#                                                           arrange_dst,
#                                                           tform)#,
#                                                           # label_associations)
#     mse_ini, l2_ini = mapali.mse_norm(src_img, dst_img, tform)
#     title_ini = 'initial (match_score:{:.2f}, mse:{:.5f}, l2:{:.2f})'.format(match_score_ini, mse_ini, l2_ini)
#     axes[0] = mapali.maplt.plot_transformed_images( images[keys[0]], images[keys[1]],
#                                               tformM=tform.params,
#                                               axes=axes[0], title=title_ini)

#     tx,ty,s,t = result['x']
#     tform_opt = skimage.transform.AffineTransform(scale=(s,s), rotation=t, translation=(tx,ty))    

#     arrange_src = copy.deepcopy(arrangements[keys[0]])
#     arrange_dst = copy.deepcopy(arrangements[keys[1]])
#     match_score_opt = mapali.arrangement_match_score_fast(arrange_src,
#                                                           arrange_dst,
#                                                           tform_opt) #,
#                                                           # label_associations)
#     mse_opt, l2_opt = mapali.mse_norm(src_img, dst_img, tform_opt)
#     title_opt = 'optimized (match_score:{:.2f}, mse:{:.5f}, l2:{:.2f})'.format(match_score_opt, mse_opt, l2_opt)
#     axes[1] = mapali.maplt.plot_transformed_images( images[keys[0]], images[keys[1]],
#                                               tformM=tform_opt.params,
#                                               axes=axes[1], title=title_opt)
    
#     # fig.savefig('{:s}_{:s}'.format(keys[0],keys[1]))
#     plt.tight_layout()
#     plt.show()

