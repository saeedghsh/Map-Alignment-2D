from __future__ import print_function

import os
import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

new_paths = [
    u'../arrangement/',
    u'../Python-CPD/',
    u'../place_categorization_2D',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import time
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
reload(arr)
# import arrangement.plotting as aplt
# reload(aplt)
import arrangement.utils as utls
reload(utls)
# import arrangement.geometricTraits as trts
# reload(trts)
import place_categorization as plcat
reload(plcat)
import map_alignment as mapali
reload(mapali)

### for Python-CPD
from functools import partial
from core import (RigidRegistration, AffineRegistration, DeformableRegistration)
# from scipy.io import loadmat


################################################################################
################################################################ functions lobby
################################################################################
def gimme():
    return None


################################################################################
####################################################################### 
################################################################################

print (4*'\t**************')
#######################################
# mapali.data_sets: is a dictionary storing file names
# mapali.data_sets.keys()
keys = ['HIH_layout', 'HIH_tango']
keys = ['kpt4a_layout', 'kpt4a_f_tango']
# keys = ['kpt4a_layout']
# keys = ['kpt4a_f_tango']
keys = ['kpt4a_layout', 'kpt4a_kb_tango']
keys = ['kpt4a_layout', 'kpt4a_kl_tango']
keys = ['kpt4a_layout', 'kpt4a_lb_tango']
# keys = ['E5_layout', 'E5_01_tango']
# keys = ['E5_layout', 'E5_02_tango']
# keys = ['E5_layout', 'E5_03_tango']
# keys = ['E5_layout', 'E5_04_tango']
# keys = ['E5_layout', 'E5_05_tango']
# keys = ['E5_layout', 'E5_06_tango']
# keys = ['E5_layout']
# keys = ['E5_06_tango']
# keys = ['E5_layout', 'E5_07_tango']
# keys = ['E5_layout', 'E5_08_tango']
keys = ['F5_layout', 'F5_01_tango']
# keys = ['F5_layout', 'F5_02_tango']
# keys = ['F5_layout', 'F5_03_tango']
# keys = ['F5_layout', 'F5_04_tango']
# keys = ['F5_layout', 'F5_05_tango']
# keys = ['E5_07_tango']
# keys = []

################################################################################
####################################################################### dev yard
################################################################################

######################################## parameters setting
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}

prun_image_occupancy_thr = 200
prun_edge_neighborhood = 5
prun_node_neighborhood = 5
prun_low_occ_percent = .025 # below "low_occ_percent"
prun_high_occ_percent = .1 # not more than "high_occ_percent"
prun_consider_categories = True

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

######################################## deployment
images = {}
label_images = {}
arrangements = {}
connectivity_maps = {}
skizs = {}
traits = {}

for key in keys:
    print ('\t *** processing map \'{:s}\':'.format(key))

    ######################################## loading file
    print ('\t loading files [image, label_image, skiz, triat_data] ...')
    image, label_image, skiz, trait = mapali.loader(mapali.data_sets[key])

    ######################################## deploying arrangement
    print ('\t deploying arrangement ... ')
    arrange = arr.Arrangement(trait, arr_config)

    ######################################## assigning place categories to faces
    print ('\t assigning place categories to faces...')
    mapali.assign_label_to_face(arrange, label_image)

    ######################################## pruning the arrangement
    print ('\t pruning arrangement wrt occupancy map ...') 
    arrange = mapali.arrangement_pruning(arrange, image,
                                         image_occupancy_thr = prun_image_occupancy_thr,
                                         edge_neighborhood = prun_edge_neighborhood,
                                         node_neighborhood = prun_node_neighborhood,
                                         low_occ_percent  = prun_low_occ_percent,
                                         high_occ_percent = prun_high_occ_percent,
                                         consider_categories = prun_consider_categories)

    ######################################## updating faces label
    # due to the changes of the arrangement and faces
    print ('\t update place categories to faces assignment ...')
    mapali.assign_label_to_face(arrange, label_image)

    ######################################## setting face attribute with shape description
    # for face matching and alignment - todo. make this a arranement method?
    for idx,face in enumerate(arrange.decomposition.faces):
        arrange.decomposition.faces[idx].set_shape_descriptor(arrange)

    ######################################## construct conectivity map
    print ('\t connectivity map construction and node profiling ...')
    mapali.set_edge_crossing_attribute(arrange, skiz,
                                       neighborhood=con_map_neighborhood,
                                       cross_thr=con_map_cross_thr)
    arrange, con_map = mapali.construct_connectivity_map(arrange, set_coordinates=True)

    # # desconnecting nodes whose corresponding faces are -1
    # discon_faces = [f_idx
    #                 for (f_idx, face) in enumerate(arrange.decomposition.faces)
    #                 if face.attributes['label_vote']==-1]
    # discon_edge = [ (n1_idx, n2_idx)
    #                 for (n1_idx, n2_idx) in con_map.edges()
    #                 if ((n1_idx in discon_faces) or (n2_idx in discon_faces)) ]
    # con_map.remove_edges_from( discon_edge )

    # profiling node, for finding label association with other maps
    con_map = mapali.profile_nodes(con_map)


    ######################################## storing results
    images[key] = image
    traits[key] = trait    
    skizs[key] = skiz 
    label_images[key] = label_image
    arrangements[key] = arrange
    connectivity_maps[key] = con_map




########## plotting 
if 1:
    row, col = 1, len(keys)
    fig, axes = plt.subplots(row, col, figsize=(20,12))
    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes] # single subplot is not a list
        
    for idx, key in enumerate(keys):

        # plotting the ogm, label_image and skiz
        mapali.plot_image(axes[idx], label_images[key], alpha=.7, cmap=None)
        mapali.plot_image(axes[idx], images[key], alpha=.5)
        mapali.plot_image(axes[idx], skizs[key], alpha=.5)
        
        # plotting face categories
        mapali.plot_place_categories(axes[idx], arrangements[key], alpha=.3)
        
        # plotting arrangement and connectivity map
        mapali.plot_arrangement(axes[idx], arrangements[key], printLabels=False)
        mapali.plot_connectivity_map(axes[idx], connectivity_maps[key])

    plt.tight_layout()
    plt.show()




######################################## Hyp-gen
print (4*'\t**************')

hyp_gen_face_similarity = ['vote','count',None][2]
hyp_gen_tform_type = ['similarity','affine'][1]
hyp_gen_enforce_match = False


#################### construct the pool of transformations
tforms = mapali.construct_transformation_population(arrangements,
                                                    connectivity_maps,
                                                    face_similarity=hyp_gen_face_similarity,
                                                    tform_type=hyp_gen_tform_type,
                                                    enforce_match=hyp_gen_enforce_match)
print ( '\t totaly {:d} transformations estimated'.format(tforms.shape[0]) )


#################### reject transformations with mismatching scale
if hyp_gen_tform_type=='affine':
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
if hyp_gen_tform_type=='affine':
    parameters = np.stack([ np.array( [tf.translation[0],
                                       tf.translation[1],
                                       tf.rotation,
                                       tf.scale[0] ] )
                            for tf in tforms ], axis=0)
elif hyp_gen_tform_type=='similarity':
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
min_s = int( .5* np.min([ len([ face
                                for face in arrangements[key].decomposition.faces
                                if face.attributes['label_vote']!=-1 ])
                          for key in keys ]) )
# min_s = int( .1* np.min([ len([ arrangements[key].decomposition.faces ])
#                           for key in keys ]) )

# min_s = 3

cls = sklearn.cluster.DBSCAN(eps=0.051, min_samples=min_s)
cls.fit(parameters)
labels = cls.labels_
unique_labels = np.unique(labels)
print ( '\t *** total: {:d} clusters...'.format(unique_labels.shape[0]-1) )
# for lbl in unique_labels:
#     print('\tcluster {:d} with {:d} memebrs'.format(lbl, len(np.nonzero(labels == lbl)[0]) ) )


#################### plotting all transformations and clusters in the "transformed unit-vector" space
if 1:
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



#################### plotting src (transformed) and dst images for the average of each cluster
if 1:
    for lbl in unique_labels:
        if lbl != -1:
            class_member_idx = np.nonzero(labels == lbl)[0]
            class_member = [ tforms[idx]
                             for idx in class_member_idx ]

            # average cluster members by parameter and create new tform
            # it's actually not that bad! inter-cluster tranforms are pretty close
            # I once (just now) checked about 200 of them...!
            t_mean = np.mean([ tf.translation for tf in class_member ], axis=0)
            if hyp_gen_tform_type=='affine':
                s_mean = np.mean([ tf.scale[0] for tf in class_member ])
            elif hyp_gen_tform_type=='similarity':
                s_mean = np.mean([ tf.scale for tf in class_member ])
            r_mean = np.mean([ tf.rotation for tf in class_member ])
            
            tf = skimage.transform.AffineTransform( scale=(s_mean,s_mean),
                                                    rotation=r_mean,
                                                    translation=t_mean)
            
            mapali.plot_trnsfromed_images(images[keys[0]], images[keys[1]], tformM=tf.params)









        

# # constructing the desired transform [E5_01]
# t = skimage.transform.AffineTransform( scale=(1.2,1.2), 
#                                        rotation=np.pi/2+0.04,
#                                        translation=(1526,15) )


# # finding targets for [E5_01] (i.e. transforms closet to desired)
# target = []
# for idx,tf in enumerate(tforms):
#     dt = np.sqrt( np.sum( (tf.translation - np.array([1526,15]))**2 ) )
#     ds = np.abs(tf.scale - 1.2)
#     dr = np.abs(tf.rotation - (np.pi/2+0.04))
#     if (ds <0.1) and (dr <0.1) and (dt <200):
#         target.append(idx)
# print (len(target))


# # checking the variance of parameters in each cluster
# for lbl in unique_labels:
#     if lbl != -1:
#         class_member_idx = np.nonzero(labels == lbl)[0]
#         class_member = [ tforms[idx]
#                          for idx in class_member_idx ]
#         t_var = np.var([ tf.translation
#                            for tf in class_member ], axis=0)
#         s_var = np.var([ tf.scale
#                            for tf in class_member ])
#         r_var = np.var([ tf.rotation
#                            for tf in class_member ])        
#         msg = 'cluster {:d}: t:({:.2f},{:.2f}) - s:{:.2f} - r:{:.2f}'
#         print (msg.format(lbl,t_var[0],t_var[1],s_var,r_var ))
# # translation has very high variance, but it's actually ok!
# # they are not that far, and the high variance seems wierd
# class_member_idx = np.nonzero(labels == 35)[0]
# class_member = [ tforms[idx]
#                  for idx in class_member_idx ]
# for tf in class_member: print (tf.translation)


# # plotting the histogram of parameters' distributions 
# fig, axes = plt.subplots(1,1, figsize=(20,12))
# axes.hist(parameters[:,0], facecolor='b', bins=100, alpha=0.7, label='tx')
# axes.hist(parameters[:,1], facecolor='r', bins=100, alpha=0.7, label='ty')
# # axes.hist(parameters[:,2], facecolor='g', bins=100, alpha=0.7, label='rotate')
# # axes.hist(parameters[:,3], facecolor='m', bins=100, alpha=0.7, label='scale')
# axes.legend(loc=1, ncol=1)
# axes.set_title('histogram of alignment parameters')
# plt.tight_layout()
# plt.show()



# # plotting all transformations and targets in 1)"transformed unit-vector" space and 2)features space (tx-ty / r-s) 
# fig, axes = plt.subplots(1,1, figsize=(20,12))
# U = np.array([1,1,1])
# # "transformed unit-vector" space
# xy = np.stack([ np.dot(tforms[idx].params, U)[:2]
#                 for idx in range(len(tforms)) ], axis=0)
# axes.plot(xy[:, 0], xy[:, 1], ',',
#           markerfacecolor='k', markeredgecolor='k')
# xy = np.stack([ np.dot(tforms[idx].params, U)[:2]
#                 for idx in target ], axis=0)
# axes.plot(xy[:, 0], xy[:, 1], '.',
#           markerfacecolor='r', markeredgecolor='r')
# # features space (tx-ty)
# xy = np.stack([ tforms[idx].translation
#                 for idx in range(len(tforms)) ], axis=0)
# axes.plot( xy[:, 0], xy[:, 1], ',',
#            markerfacecolor='k', markeredgecolor='k')
# xy = np.stack([ tforms[idx].translation
#                 for idx in target ], axis=0)
# axes.plot( xy[:, 0], xy[:, 1], '.',
#            markerfacecolor='r', markeredgecolor='r')
# # features space (r-s)
# xy = np.stack([ (tforms[idx].scale, tforms[idx].rotation)
#                 for idx in range(len(tforms)) ], axis=0)
# axes.plot( xy[:, 0], xy[:, 1], ',',
#            markerfacecolor='k', markeredgecolor='k')
# xy = np.stack([ (tforms[idx].scale, tforms[idx].rotation)
#                 for idx in target ], axis=0)
# axes.plot( xy[:, 0], xy[:, 1], '.',
#            markerfacecolor='r', markeredgecolor='r')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()











################################################################################
###################################################### CPD: coherent point drift
################################################################################
'''
What to use? nodes from arrangement.prime or point samples from occupancy?
if using occupancy, I can also try daniels work with distance image from skiz computation
'''

# # src: tango
# # dst: layout
# rotate_src = [np.pi, False][0]
# point_set = ['arrangement', 'point_cloud'][0]

# if point_set == 'arrangement':
#     src = [arrange2.graph.node[key]['obj'].point for key in arrange2.graph.node.keys()]
#     src = np.array([ [float(p.x),float(p.y)] for p in src ])
#     dst = [arrange1.graph.node[key]['obj'].point for key in arrange1.graph.node.keys()]
#     dst = np.array([ [float(p.x),float(p.y)] for p in dst ])

# elif point_set == 'point_cloud':
#     # nonzero returns ([y,..],[x,...])
#     # flipped to have (x,y)
#     skip = 5
#     dst = np.fliplr( np.transpose(np.nonzero(img1<50)) )[0:-1:skip]
#     src = np.fliplr( np.transpose(np.nonzero(img2<30)) )[0:-1:skip]

# #### plotting before rotate
# mapali.plot_point_sets (src, dst)

# ### transforming before registeration
# if rotate_src:
#     # np.dot(R,V) - rotating an standing vector V with the rotation matrix R
#     # np.dot(S,R.T) - rotating a point set S (nx2) with the rotation matrix R
#     t = rotate_src
#     R = np.array([ [np.cos(t), -np.sin(t)],
#                    [np.sin(t),  np.cos(t)] ])
#     src = np.dot(src, R.T)
#     mapali.plot_point_sets (src, dst)

# fig = plt.figure()
# fig.add_axes([0, 0, 1, 1])
# callback = partial(mapali.visualize, ax=fig.axes[0])
# reg = RigidRegistration(dst, src, sigma2=None, maxIterations=100, tolerance=0.001)
# # reg = AffineRegistration(dst, src)
# # reg = DeformableRegistration(dst, src)
# Y_transformed, s, R, t = reg.register(callback)
# plt.show()
# print (reg.err)
