from __future__ import print_function

import os
import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

new_paths = [
    u'../arrangement/',
    # u'/home/saesha/Desktop/arrangement',
    u'../Python-CPD/',
    u'../place_categorization_2D',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

# sys.path.append( u'/usr/share/inkscape/extensions' )
# import inkex

import time
import copy
import itertools 
import operator
import cv2
import numpy as np
import sympy as sym
import scipy
import networkx as nx
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt

import arrangement.arrangement as arr
# reload(arr)
# import arrangement.plotting as aplt
# reload(aplt)
# import arrangement.utils as utls
# reload(utls)
# import arrangement.geometricTraits as trts
# reload(trts)
# import place_categorization as plcat
# reload(plcat)
import map_alignment as mapali
# reload(mapali)

### for Python-CPD
from functools import partial
import core as PCD
# reload (PCD)
# from core import (RigidRegistration, AffineRegistration, DeformableRegistration)
# from scipy.io import loadmat


################################################################################
################################################################ functions lobby
################################################################################
def gimme():
    '''{:s}'''.format('yep')
    return None

################################################################################
####################################################################### 
################################################################################

print (4*'\t**************')
#######################################
# mapali.data_sets: is a dictionary storing file names
# mapali.data_sets.keys()
keys = ['HIH_layout']
keys = [ ['HIH_layout', 'HIH_tango'], ['HIH_tango'] ][0]
# keys = ['kpt4a_layout']
keys = [ ['kpt4a_layout', 'kpt4a_f_tango'], ['kpt4a_f_tango'] ] [0]
keys = [ ['kpt4a_layout', 'kpt4a_kb_tango'], ['kpt4a_kb_tango'] ] [0]
keys = [ ['kpt4a_layout', 'kpt4a_kl_tango'], ['kpt4a_kl_tango'] ] [0]
keys = [ ['kpt4a_layout', 'kpt4a_lb_tango'], ['kpt4a_lb_tango'] ] [0]
# keys = ['E5_layout']
# keys = [ ['E5_layout', 'E5_01_tango'], ['E5_01_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_02_tango'], ['E5_02_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_03_tango'], ['E5_03_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_04_tango'], ['E5_04_tango'] ] [1]
keys = [ ['E5_layout', 'E5_05_tango'], ['E5_05_tango'] ] [0]
# keys = [ ['E5_layout', 'E5_06_tango'], ['E5_06_tango'] ] [0]
keys = [ ['E5_layout', 'E5_07_tango'], ['E5_07_tango'] ] [0]
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

prun_image_occupancy_thr = 200
prun_edge_neighborhood = 5
prun_node_neighborhood = 5
prun_low_occ_percent = .025 # below "low_occ_percent"
# prun_low_occ_percent = .0125 # below "low_occ_percent"
prun_high_occ_percent = .1 # not more than "high_occ_percent"
prun_consider_categories = [True, False][0]


face_low_occ_percent = .05 # self below "low_occ_percent"
face_high_occ_percent = .1 # no nodes more than "high_occ_percent"
face_consider_categories = [True, False][0]
face_similar_thr = 0.4

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

######################################## deployment
images = {}
label_images = {}
dis_images = {}
skizs = {}
traits = {}

arrangements = {}
connectivity_maps = {}

for key in keys:
    print ('\t *** processing map \'{:s}\':'.format(key))

    ######################################## loading file
    print ('\t loading files [image, label_image, skiz, triat_data] ...')
    image, label_image, dis_image, skiz, trait = mapali.loader(mapali.data_sets[key])

    ######################################## deploying arrangement
    print ('\t deploying arrangement ... ')
    arrange = arr.Arrangement(trait, arr_config)

    ######################################## assigning place categories to faces
    print ('\t assigning place categories to faces...')
    mapali.assign_label_to_all_faces(arrange, label_image)

    ######################################## edge pruning the arrangement
    print ('\t edge pruning arrangement wrt occupancy map ...') 
    arrange = mapali.prune_arrangement( arrange, image,
                                        image_occupancy_thr = prun_image_occupancy_thr,
                                        edge_neighborhood = prun_edge_neighborhood,
                                        node_neighborhood = prun_node_neighborhood,
                                        low_occ_percent  = prun_low_occ_percent,
                                        high_occ_percent = prun_high_occ_percent,
                                        consider_categories = prun_consider_categories)

    ######################################## updating faces label
    # due to the changes of the arrangement and faces
    print ('\t update place categories to faces assignment ...')
    mapali.assign_label_to_all_faces(arrange, label_image)    

    # ######################################## face growing the arrangement
    # print ('\t face growing the ...') 
    # arrange = mapali.prune_arrangement_with_face_growing (arrange, label_image,
    #                                         low_occ_percent=face_low_occ_percent,
    #                                         high_occ_percent=face_high_occ_percent,
    #                                         consider_categories=face_consider_categories,
    #                                         similar_thr=face_similar_thr)

    # ######################################## updating faces label
    # # due to the changes of the arrangement and faces
    # print ('\t update place categories to faces assignment ...')
    # mapali.assign_label_to_all_faces(arrangements[key], label_images[key])    

    ######################################## setting face attribute with shape description
    # for face matching and alignment - todo. make this a arranement method?
    print ('\t setting face attribute with shape description ...')    
    for idx,face in enumerate(arrange.decomposition.faces):
        arrange.decomposition.faces[idx].set_shape_descriptor(arrange)


    ######################################## construct conectivity map
    print ('\t connectivity map construction and node profiling ...')
    mapali.set_edge_crossing_attribute(arrange, skiz,
                                       neighborhood=con_map_neighborhood,
                                       cross_thr=con_map_cross_thr)
    arrange, con_map = mapali.construct_connectivity_map(arrange, set_coordinates=True)

    # profiling node, for finding label association with other maps
    con_map = mapali.profile_nodes(con_map)


    ######################################## storing results
    images[key] = image
    label_images[key] = label_image
    dis_images[key] = dis_image
    skizs[key] = skiz
    traits[key] = trait
    arrangements[key] = arrange
    connectivity_maps[key] = con_map


########## plotting 
if 0:
    row, col = 1, len(keys)
    fig, axes = plt.subplots(row, col, figsize=(20,12))
    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes] # single subplot is not a list
        
    for idx, key in enumerate(keys):

        ### plotting the ogm, label_image and skiz
        mapali.plot_image(axes[idx], label_images[key], alpha=.7, cmap=None)
        mapali.plot_image(axes[idx], images[key], alpha=.5)
        # mapali.plot_image(axes[idx], skizs[key], alpha=.5)
        
        ### plotting arrangement and connectivity map
        mapali.plot_arrangement(axes[idx], arrangements[key], printLabels=False)
        # mapali.plot_connectivity_map(axes[idx], connectivity_maps[key])

        ### plotting face categories
        # mapali.plot_place_categories(axes[idx], arrangements[key], alpha=.3)

        ### plot edge occupancy percent - text
        # mapali.plot_text_edge_occupancy(axes[idx], arrangements[key])
        

    plt.tight_layout()
    plt.show()







################################################################################
########################################################## Hypothesis generation
################################################################################
print (4*'\t**************')


hypgen_face_similarity = ['vote','count',None][2]
hypgen_tform_type = ['similarity','affine'][1]
hypgen_enforce_match = False

print (4*'\t**************')
print ('\t check this out: ')
print ('\t use [Oriented Minimum Bounding Box] of faces instead of faces themselves ')
print ('\t for hypothesis generation')
print (4*'\t**************')


#################### construct the pool of transformations
tforms = mapali.construct_transformation_population(arrangements,
                                                    connectivity_maps,
                                                    face_similarity=hypgen_face_similarity,
                                                    tform_type=hypgen_tform_type,
                                                    enforce_match=hypgen_enforce_match)
print ( '\t totaly {:d} transformations estimated'.format(tforms.shape[0]) )


#################### reject transformations with mismatching scale
if hypgen_tform_type=='affine':
    matching_scale_idx = []
    for idx,tf in enumerate(tforms):
        if np.abs(tf.scale[0]-tf.scale[1])/np.min(tf.scale) < .1: # >.05
            matching_scale_idx.append(idx)
    print ( '\t {:d} tforms rejected due to scale mismatch'.format( len(tforms)-len(matching_scale_idx) ) )
    tforms = tforms[matching_scale_idx]


#################### reject transformations, if images won't overlap
if 1:
    src_h, src_w = images[keys[0]].shape
    src = np.array([ [0,0], [src_w,0], [src_w,src_h], [0,src_h], [0,0] ])
    dst_h, dst_w = images[keys[1]].shape
    dst = np.array([ [0,0], [dst_w,0], [dst_w,dst_h], [0,dst_h], [0,0] ])
    dst_path = mapali.create_mpath(dst)
    
    overlapping_idx = []
    for idx,tf in enumerate(tforms):
        src_warp = np.dot(tf.params, np.hstack( (src, np.array([[1],[1],[1],[1],[1]]))).T ).T[:,:-1]
        src_warp_path = mapali.create_mpath(src_warp)
        if src_warp_path.intersects_path(dst_path,filled=True):
            overlapping_idx.append(idx)
    print ('\t {:d} tforms rejected due to non-overlapping'.format( len(tforms)-len(overlapping_idx) ))
    
    tforms = tforms[overlapping_idx]


#################### extracting parameters of the transforms for clustering
# using parameters to represent alignments
if hypgen_tform_type=='affine':
    parameters = np.stack([ np.array( [tf.translation[0],
                                       tf.translation[1],
                                       tf.rotation,
                                       tf.scale[0] ] )
                            for tf in tforms ], axis=0)
elif hypgen_tform_type=='similarity':
    parameters = np.stack([ np.array( [tf.translation[0],
                                       tf.translation[1],
                                       tf.rotation,
                                       tf.scale ] )
                            for tf in tforms ], axis=0)


####################  rejecting transformations with NaN - it happens very often
row_idx, col_idx = np.nonzero( np.isnan(parameters)==False )
print ('\t {:d} transforms rejected for containng NaN'.format(parameters.shape[0]-len(np.unique(row_idx))))
parameters = parameters[ np.unique(row_idx), :]
tforms = tforms[ np.unique(row_idx) ]


####################  rejecting weird transforms
if 1:
    idx = np.arange(parameters.shape[0])
    # print (idx.shape)
    # rejecting transforms with translation > 10**5
    # translate_threshold = np.max([np.sqrt(image[key].shape[0]**2 + image[key].shape[0]**2) for key in keys])
    translate_threshold = 10000
    idx_tmp = np.nonzero( np.abs(parameters[:,0])< translate_threshold )[0]
    idx = np.intersect1d(idx, idx_tmp)
    idx_tmp = np.nonzero( np.abs(parameters[:,1])< translate_threshold )[0]
    idx = np.intersect1d(idx, idx_tmp)
    # print (idx.shape)
    
    # rejecting transforms with scale < 0.1 or scale > 10
    idx_tmp = np.nonzero( 0.1 < parameters[:,3] )[0]
    idx = np.intersect1d(idx, idx_tmp)
    idx_tmp = np.nonzero( parameters[:,3] < 10 )[0]
    idx = np.intersect1d(idx, idx_tmp)
    # print (idx.shape)

    print ('\t {:d} weird transforms rejected'.format(parameters.shape[0]-idx.shape[0]))
    parameters = parameters[ idx,: ]
    tforms = tforms[ idx ]



#################### feature scaling (standardization)
# note that the scaling is only applied to parametes (for clustering), not the transformations themselves
parameters -= np.mean( parameters, axis=0 )
parameters /= np.std( parameters, axis=0 )


#################### clustering pool into hypotheses
import sklearn.cluster
print ('\t clustering {:d} transformations'.format(parameters.shape[0]))
# min_s = int( .5* np.min([ len([ face
#                                 for face in arrangements[key].decomposition.faces
#                                 if face.attributes['label_vote']!=-1 ])
#                           for key in keys ]) )

min_s = int( .1* np.min([ len([ arrangements[key].decomposition.faces ])
                          for key in keys ]) )

min_s = 4
print (min_s)

cls = sklearn.cluster.DBSCAN(eps=0.051, min_samples=min_s)
cls.fit(parameters)
labels = cls.labels_
unique_labels = np.unique(labels)
print ( '\t *** total: {:d} clusters...'.format(unique_labels.shape[0]-1) )
# for lbl in unique_labels:
#     print('\tcluster {:d} with {:d} memebrs'.format(lbl, len(np.nonzero(labels == lbl)[0]) ) )


#################### plotting all transformations and clusters
#################### in the "transformed unit-vector" space
if 0:
    fig, axes = plt.subplots(1,1, figsize=(20,12))
    U = np.array([1,1,1])
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for lbl, col in zip(unique_labels, colors):
        
        mrk = '.'
        if lbl == -1: col, mrk = 'k', ','
        
        class_member_idx = np.nonzero(labels == lbl)[0]
        xy = np.stack([ np.dot(tforms[idx].params, U)[:2]
                        for idx in class_member_idx ], axis=0)
        axes.plot( xy[:, 0], xy[:, 1], mrk,
                   markerfacecolor=col, markeredgecolor=col)
        
    # # plotting predefined target cluster
    # xy = np.stack([ np.dot(tforms[idx].params, U)[:2]
    #                 for idx in target ], axis=0)
    # axes.plot(xy[:, 0], xy[:, 1], '*',
    #           markerfacecolor='r', markeredgecolor='r')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()



#################### plotting src (transformed) and dst images
#################### for the average of each cluster
if 0:
    for lbl in unique_labels:
        if lbl != -1:
            class_member_idx = np.nonzero(labels == lbl)[0]
            class_member = [ tforms[idx]
                             for idx in class_member_idx ]

            # average cluster members by parameter and create new tform
            # it's actually not that bad! inter-cluster tranforms are pretty close
            # I once (just now) checked about 200 of them...!
            t_mean = np.mean([ tf.translation for tf in class_member ], axis=0)
            if hypgen_tform_type=='affine':
                s_mean = np.mean([ tf.scale[0] for tf in class_member ])
            elif hypgen_tform_type=='similarity':
                s_mean = np.mean([ tf.scale for tf in class_member ])
            r_mean = np.mean([ tf.rotation for tf in class_member ])
            
            tf = skimage.transform.AffineTransform( scale=(s_mean,s_mean),
                                                    rotation=r_mean,
                                                    translation=t_mean)
            
            mapali.plot_transformed_images(images[keys[0]],
                                           images[keys[1]],
                                           tformM=tf.params,
                                           title='cluster {:d}'.format(lbl))


# ### plotting the tforms of the "good cluster"
# good_cluster_idx = 3  # E5_05
# good_cluster_idx = 14 # E5_07
# good_cluster = [ tforms[idx] for idx in np.nonzero(labels == good_cluster_idx)[0] ]
# hypotheses = [tf in good_cluster]

# for tf_idx,tf in enumerate(good_cluster):
#     title = 'cluster {:d} - element {:d} '.format(good_cluster_idx, tf_idx)
#     mapali.plot_transformed_images(images[keys[0]],
#                                    images[keys[1]],
#                                    tformM=tf.params,
#                                    title=title)


############################# pick the winning hypothesis - face 2 face association and match score
arr_match_score = {}
for lbl in unique_labels:
    if lbl != -1:
        print (lbl)
        class_member_idx = np.nonzero(labels == lbl)[0]
        class_member = [ tforms[idx]
                         for idx in class_member_idx ]

        t_mean = np.mean([ tf.translation for tf in class_member ], axis=0)
        r_mean = np.mean([ tf.rotation for tf in class_member ])
        if hypgen_tform_type=='affine':
            s_mean = np.mean([ tf.scale[0] for tf in class_member ])
        elif hypgen_tform_type=='similarity':
            s_mean = np.mean([ tf.scale for tf in class_member ])
            
        tf = skimage.transform.AffineTransform( scale=(s_mean,s_mean),
                                                rotation=r_mean,
                                                translation=t_mean)

        arrange_src = copy.deepcopy(arrangements[keys[0]])
        arrange_dst = copy.deepcopy(arrangements[keys[1]])
        arr_match_score[lbl] = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf)


good_cluster_idx = max(arr_match_score, key=arr_match_score.get)
good_cluster = [ tforms[idx] for idx in np.nonzero(labels == good_cluster_idx)[0] ]
# hypotheses = [tf in good_cluster]

arr_match_score = {}
for idx, tf in enumerate(good_cluster):
    arrange_src = copy.deepcopy(arrangements[keys[0]])
    arrange_dst = copy.deepcopy(arrangements[keys[1]])
    arr_match_score[idx] = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf)

for k in arr_match_score:
    plt.plot(k, arr_match_score[k], 'ro')
plt.show()

best_idx = max(arr_match_score, key=arr_match_score.get)
hypothesis =  good_cluster[best_idx]


mapali.plot_transformed_images(images[keys[0]], images[keys[1]],
                               tformM= hypothesis.params,
                               title='winner is the element {:d} from cluster {:d}'.format(best_idx, good_cluster_idx))


########################################
######## optimizing with distance images
########################################
tf = hypothesis

if 1:
    src_img = dis_images[keys[0]]
    dst_img = dis_images[keys[1]]
else:
    thr1,thr2 = [120, 255]
    img = np.flipud( cv2.imread( mapali.data_sets[keys[0]], cv2.IMREAD_GRAYSCALE) )
    ret, src_img = cv2.threshold(img.astype(np.uint8) , thr1,thr2 , cv2.THRESH_BINARY)
    img = np.flipud( cv2.imread( mapali.data_sets[keys[1]], cv2.IMREAD_GRAYSCALE) )
    ret, dst_img = cv2.threshold(img.astype(np.uint8) , thr1,thr2 , cv2.THRESH_BINARY)

X0 = (tf.translation[0], tf.translation[1], tf.scale[0], tf.rotation)
X_bounds = ((None,None),(None,None),(None,None),(None,None)) # No bounds
X_bounds = ((X0[0]-100,X0[0]+100),(X0[1]-100,X0[1]+100),
            (X0[2]-.1, X0[2]+.1), (X0[3]-.08,X0[3]+.08))
methods = [ 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
            'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']#[1,5,6,7,8,]
# [4,9,10]: need jac
# [0,2,3]: did not converge

result = scipy.optimize.minimize( mapali.objectivefun_image, X0,
                                  args=(src_img, dst_img),
                                  method = methods[1],
                                  # bounds = X_bounds,
                                  tol=1e-6,
                                  options={'maxiter':100, 'disp':True} )

if result['success']:
    fig, axes = plt.subplots(1,2, figsize=(20,12))
    
    arrange_src = copy.deepcopy(arrangements[keys[0]])
    arrange_dst = copy.deepcopy(arrangements[keys[1]])
    match_score_ini = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf)
    mse_ini, l2_ini = mapali.mse_norm(src_img, dst_img, tf)
    title_ini = 'initial (match_score:{:.2f}, mse:{:.5f}, l2:{:.2f})'.format(match_score_ini, mse_ini, l2_ini)
    axes[0] = mapali.plot_transformed_images( images[keys[0]], images[keys[1]],
                                              tformM=tf.params,
                                              axes=axes[0], title=title_ini)

    tx,ty,s,t = result['x']
    tf_opt = skimage.transform.AffineTransform(scale=(s,s), rotation=t, translation=(tx,ty))    

    arrange_src = copy.deepcopy(arrangements[keys[0]])
    arrange_dst = copy.deepcopy(arrangements[keys[1]])
    match_score_opt = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf_opt)
    mse_opt, l2_opt = mapali.mse_norm(src_img, dst_img, tf_opt)
    title_opt = 'optimized (match_score:{:.2f}, mse:{:.5f}, l2:{:.2f})'.format(match_score_opt, mse_opt, l2_opt)
    axes[1] = mapali.plot_transformed_images( images[keys[0]], images[keys[1]],
                                              tformM=tf_opt.params,
                                              axes=axes[1], title=title_opt)
    plt.tight_layout()
    plt.show()

