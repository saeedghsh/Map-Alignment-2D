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
import numpy as np
import cv2
import skimage.transform
import sympy as sym
import scipy.ndimage
import sklearn.cluster

import matplotlib
import matplotlib.pyplot as plt

import multiprocessing as mp
import contextlib as ctx
from functools import partial

from map_alignment import map_alignment as mapali
from map_alignment import utilities as utils
# reload(mapali)
# reload(mapali.maplt)
# reload(utils)

################################################################################
############################################################### Functions' Lobby
################################################################################
def _find_dominant_orientations(image,
                                orthogonal_orientations=False,
                                find_peak_conf={} ):
    '''
    intense smoothing of the image and weithed histogram of gradient is very
    cruicial, min value for filter size for image processing should be 9
    and for 1D signal smoothing atleast 11
    
    '''
    # flipud: why? I'm sure it won't work otherwise, but dont know why
    # It should happen in both "_find_dominant_orientations" & "find_grid_lines"
    img = np.flipud(image)

    ### smoothing
    img = cv2.blur(img, (11,11))
    img = cv2.GaussianBlur(img, (11,11),0)    
    
    ### oriented gradient of the image
    # is it (dx - 1j*dy) or (dx + 1j*dy)
    # this is related to "flipud" problem mentioned above
    dx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize=11)
    dy = cv2.Sobel(img, cv2.CV_64F, 0,1, ksize=11)
    grd = dx - 1j*dy
    # return grd 

    
    ### weighted histogram of oriented gradients (over the whole image)
    hist, binc = utils.wHOG (grd, NumBin=180*5, Extension=False)
    hist = utils.smooth(hist, window_len=21)
    # return hist, binc

    ### finding peaks in the histogram
    if orthogonal_orientations:
        # if dominant orientations are orthogonal, find only one and add pi/2 to it
        peak_idx = np.argmax(hist)
        orientations = [binc[peak_idx], binc[peak_idx]+np.pi/2]

    else:
        # setting cwt_peak_detection parameters
        CWT = True if ('CWT' not in find_peak_conf) else find_peak_conf['CWT']
        cwt_range = (5,50,5) if ('cwt_range' not in find_peak_conf) else find_peak_conf['cwt_range']
        Refine_win = 20 if ('Refine_win' not in find_peak_conf) else find_peak_conf['Refine_win']
        MinPeakDist = 30 if ('MinPeakDist' not in find_peak_conf) else find_peak_conf['MinPeakDist']
        MinPeakVal = .2 if ('MinPeakVal' not in find_peak_conf) else find_peak_conf['MinPeakVal']
        Polar = False if ('Polar' not in find_peak_conf) else find_peak_conf['Polar']
        
        peak_idx = utils.FindPeaks( hist,
                                    CWT=True, cwt_range=(5,50,5),
                                    Refine_win=20 , MinPeakDist = 30 , MinPeakVal=.2,
                                    Polar=False )
        
        orientations = list(binc[peak_idx])
    
    # shrinking the range to [-pi/2, pi/2]
    for idx in range(len(orientations)):
        if orientations[idx]< -np.pi/2:
            orientations[idx] += np.pi
        elif np.pi/2 <= orientations[idx]:
            orientations[idx] -= np.pi

    # removing similar angles
    for idx in range(len(orientations)-1,-1,-1):
        for jdx in range(idx):
            if np.abs(orientations[idx] - orientations[jdx]) < np.spacing(10**10):
                orientations.pop(idx)
                break

    ### returning the results
    return np.array(orientations)

################################################################################
def _find_lines_with_radiography(image, orientations, peak_detect_config):
    '''
    '''
    if len(orientations)==0: raise(NameError('No dominant orientation is available'))
    if peak_detect_config is None: peak_detect_config = [10, 15, 0.15]

    ### fetching setting for sinogram peak detection
    [refWin, minDist, minVal] = peak_detect_config
    
    ### radiography
    # flipud: why? I'm sure it won't work otherwise, but dont know why
    # It should happen in both "_find_dominant_orientations" & "find_grid_lines"
    image = np.flipud(image)
    imgcenter = (image.shape[1]/2, # cols == x
                 image.shape[0]/2) # rows == y
    
    sinog_angles = orientations - np.pi/2 # in radian
    sinograms = skimage.transform.radon(image, theta=sinog_angles*180/np.pi )#, circle=True)
    sinogram_center = len(sinograms.T[0])/2
    
    lines = []
    for (orientation, sinog_angle, sinogram) in zip(orientations, sinog_angles, sinograms.T):
        # Find peaks in sinogram:
        peakind = utils.FindPeaks(utils.smooth(sinogram, window_len=11),
                                  CWT=False, cwt_range=(1,8,1),
                                  Refine_win = int(refWin),
                                  MinPeakDist = int(minDist),
                                  MinPeakVal = minVal,
                                  Polar=False)
            
        # line's distance to the center of the image
        dist = np.array(peakind) - sinogram_center
        
        pts_0 = [ ( imgcenter[0] + d*np.cos(sinog_angle),
                    imgcenter[1] + d*np.sin(sinog_angle) )
                  for d in dist]
        
        pts_1 = [ ( point[0] + np.cos(orientation) ,
                    point[1] + np.sin(orientation) )
                  for point in pts_0]
        
        lines += [mapali.arr.trts.LineModified( args=( sym.Point(p0[0],p0[1]), sym.Point(p1[0],p1[1]) ) )
                  for (p0,p1) in zip(pts_0,pts_1)]

    return lines

################################################################################
def _lock_n_load(file_name, config={}):
    '''
    # loading image:
    loads the image pointed to by the file_name

    NOTE on binarization of image, it happens twice:
    1) input for SKIZ and distance tranforms; where unexplored areas are treated
    as occupied area, because we are only interested SKIZ and distance transform
    of open space.
    2) trait detection; unexplored areas are treated as occupied area, because
    we are only interested in occupied cells for line detection.
    
    results: 'image', 'traits', 'skiz', 'distance'
    '''
    ### default values
    if 'trait_detection_source' not in config: config['trait_detection_source'] = 'binary_inverted'
    if 'binary_thresholding_1' not in config: config['binary_thresholding_1'] = 200
    if 'binary_thresholding_2' not in config: config['binary_thresholding_2'] = [100, 255]
    if 'edge_detection_config' not in config: config['edge_detection_config'] = [50, 150, 3]
    if 'peak_detect_sinogram_config' not in config: config['peak_detect_sinogram_config'] = [10, 15, 0.15]
    if 'orthogonal_orientations' not in config: config['orthogonal_orientations'] = True
    results = {}
    
    ######################################## laoding image
    img = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )
    results['image'] = img

    tic = time.time()

    ######################################## get skiz and distance image
    ## NOTE: this is the first binary conversion (see documentation of the method)
    thr = config['binary_thresholding_1']
    bin_ = np.array( np.where( img < thr, 0, 255 ), dtype=np.uint8)
    img_skiz, img_disance = mapali.skiz_bitmap(bin_, invert=True, return_distance=True)
    
    # scaling distance image to [0, 255]
    img_disance += img_disance.min()
    img_disance *= 255. / img_disance.max()
    
    results['skiz'] = img_skiz
    results['distance'] = img_disance

    ######################################## detect geometric traits
    if 'trait_file_name' in config:
        trait_file_name = config['trait_file_name']

        file_ext = trait_file_name.split('.')[-1]
        if file_ext in ['yaml', 'YAML', 'yml', 'YML']:
            trait_data = mapali.arr.utls.load_data_from_yaml( trait_file_name )   
            traits = trait_data['traits']
        else:
            raise(NameError('loading traits from file only supports yaml (svg to come soon)'))

    else:

        ########## detect dominant orientation
        orientations = _find_dominant_orientations(img, config['orthogonal_orientations'])

        ########## get the appropriate input image
        if 'trait_detection_source' not in config:
            print ('WARNING: \'trait_detection_source\' is not defined, origingal is selected')
            image = img

        elif config['trait_detection_source'] == 'original':
            image = img

        elif config['trait_detection_source'] == 'binary':
            ## NOTE: this is the second binary conversion (see documentation of the method)
            [thr1, thr2] = config['binary_thresholding_2']
            ret, bin_img = cv2.threshold(img, thr1,thr2 , cv2.THRESH_BINARY)
            image = bin_img

        elif config['trait_detection_source'] == 'binary_inverted':
            ## NOTE: this is the second binary conversion (see documentation of the method)
            [thr1, thr2] = config['binary_thresholding_2']
            ret, bin_img = cv2.threshold(img , thr1,thr2 , cv2.THRESH_BINARY_INV)
            image = bin_img
            
        elif config['trait_detection_source'] == 'edge':
            edge_img = cv2.Canny(img, thr1, thr2, apt_size)
            image = edge_img
    
        traits = _find_lines_with_radiography(image, orientations, config['peak_detect_sinogram_config'])


    ########## triming lines to segments, based on boundary
    boundary = [-20, -20, image.shape[1]+20, image.shape[0]+20]
    traits = mapali.arr.utls.unbound_traits(traits)
    traits = mapali.arr.utls.bound_traits(traits, boundary)
    results['traits'] = traits
    
    elapsed_time = time.time() - tic

    return results, elapsed_time


################################################################################
def _construct_arrangement(data, config={}):
    '''
    '''
    ########## setting default values

    if 'multi_processing' not in config: config['multi_processing'] = 4
    if 'end_point' not in config: config['end_point'] = False
    if 'timing' not in config: config['timing'] = False

    if 'prune_dis_neighborhood' not in config: config['prune_dis_neighborhood'] = 2
    if 'prune_dis_threshold' not in config: config['prune_dis_threshold'] = .15 # home:0.15 - office:0.075

    # any cell with value below this is considered occupied
    if 'occupancy_threshold' not in config: config['occupancy_threshold'] = 200


    tic = time.time()
    ######################################## deploying arrangement
    # if print_messages: print ('\t deploying arrangement ... ')
    # tic_ = time.time()
    arrange = mapali.arr.Arrangement(data['traits'], config)
    # print ('arrangment time:{:.5f}'.format(time.time()-tic_))

    ###############  distance based edge pruning
    # if print_messages: print ('\t arrangement pruning ... ')
    # tic_= time.time()
    mapali.set_edge_distance_value(arrange, data['distance'], config['prune_dis_neighborhood'])
    # print ('set_edge_distance_value:{:.5f}'.format(time.time()-tic_))
    # tic_= time.time()
    arrange = mapali.prune_arrangement_with_distance(arrange, data['distance'],
                                                     neighborhood=config['prune_dis_neighborhood'],
                                                     distance_threshold=config['prune_dis_threshold'])
    # print ('arrangment pruning:{:.5f}'.format(time.time()-tic_))

    ############### counting occupied cells in each face
    # if print_messages: print ('\t counting occupancy of faces ... ')
    # tic_= time.time()
    for face in arrange.decomposition.faces:
        pixels_in_path = mapali.get_pixels_in_mpath(face.path, image_shape=data['image'].shape)
        pixels_val = data['image'][pixels_in_path[:,1],pixels_in_path[:,0]]
        total = float(pixels_val.shape[0])
        occupied = float(np.count_nonzero(pixels_val < config['occupancy_threshold']))
        face.attributes['occupancy'] = [occupied, total]
    # print ('counting occupancy of faces:{:.5f}'.format(time.time()-tic_))

    ######################################## 
    # if print_messages: print ('\t setting ombb attribute of faces ...') 
    # tic_ = time.time()
    arrange = mapali.set_ombb_of_faces (arrange)
    # print ('setting ombb attribute of faces:{:.5f}'.format(time.time()-tic_))    
    ######################################## 
    # if print_messages: print ('\t caching face area weight ...') 
    # tic_ = time.time()    
    superface = arrange._get_independent_superfaces()[0]
    arrange_area = superface.get_area()
    for face in arrange.decomposition.faces:
        face.attributes['area_weight'] = float(face.get_area()) / float(arrange_area)
    # print ('face area weight:{:.5f}'.format(time.time()-tic_))    

    elapsed_time = time.time() - tic
    return arrange, elapsed_time

################################################################################
def _generate_hypothese(src_arr, src_img_shape,
                       dst_arr, dst_img_shape,
                       config={}):
    '''
    finds transformations between mbb of faces that are not outliers
    similar_labels (default: ignore)
    if similar_labels to be used, "connectivity_maps" should be passed here

    parameters for "align_ombb()":
    - tform_type='affine'

    parameters for "reject_implausible_transformations()":
    - scale_mismatch_ratio_threshold (default: 0.5)    
    - scale_bounds (default: [.3, 3])

    '''

    if 'scale_mismatch_ratio_threshold' not in config: config['scale_mismatch_ratio_threshold'] = .5
    if 'scale_bounds' not in config: config['scale_bounds'] = [.3, 3] #[.1, 10]

    if 'face_occupancy_threshold' not in config: config['face_occupancy_threshold'] = .5
    # if '' not in config: config[''] = 

    tic = time.time()

    tforms = []
    for face_src in src_arr.decomposition.faces:
        # check if src_face is not occupied
        src_occ, src_tot = face_src.attributes['occupancy']
        if (src_occ/src_tot) < config['face_occupancy_threshold']:
            for face_dst in dst_arr.decomposition.faces:
                # check if dst_face is not occupied
                dst_occ, dst_tot = face_dst.attributes['occupancy']
                if (dst_occ/dst_tot) < config['face_occupancy_threshold']:
                    tforms.extend (mapali.align_ombb(face_src,face_dst, tform_type='affine'))

    tforms = np.array(tforms)
    if print_messages: print ( '\t totaly {:d} transformations estimated'.format(tforms.shape[0]) )
    tforms_total = tforms.shape[0]
    
    tforms = mapali.reject_implausible_transformations( tforms,
                                                        src_img_shape, dst_img_shape,
                                                        config['scale_mismatch_ratio_threshold'],
                                                        config['scale_bounds'] )
    elapsed_time = time.time() - tic
    tforms_after_reject = tforms.shape[0]

    if print_messages: print ( '\t and {:d} transformations survived the rejections...'.format(tforms.shape[0]) )
    # if tforms.shape[0] == 0: raise (NameError('no transformation survived.... '))

    return tforms, elapsed_time, tforms_total, tforms_after_reject # , (total_t, after_reject)


################################################################################
def _select_winning_hypothesis(src_arr, dst_arr, tforms, config={}):
    '''
    input:
    arrangements, tforms
    
    too_many_tforms = 1000
    sklearn.cluster.DBSCAN(eps=0.051, min_samples=2)
    
    output:
    hypothesis
    '''

    if 'multiprocessing' not in config: config['multiprocessing'] = True
    if 'too_many_tforms' not in config: config['too_many_tforms'] = 1000
    if 'dbscan_eps' not in config: config['dbscan_eps'] = 0.051
    if 'dbscan_min_samples' not in config: config['dbscan_min_samples'] = 2
    # if '' not in config: config[''] = 

    tic = time.time()

    if tforms.shape[0] < config['too_many_tforms']:
        if print_messages: print ('only {:d} tforms are estimated, so no clustering'.format(tforms.shape[0]))

        if config['multiprocessing']:
            arrangement_match_score_par = partial( _arrangement_match_score_4partial,
                                                   arrangement_src=src_arr,
                                                   arrangement_dst=dst_arr,
                                                   tforms = tforms)
                        
            with ctx.closing(mp.Pool(processes=4)) as p:
                arr_match_score = p.map( arrangement_match_score_par, range(len(tforms)))

            best_idx = np.argmax(arr_match_score)

        else:
            arr_match_score = {}
            for idx, tf in enumerate(tforms):
                arrange_src = src_arr
                arrange_dst = dst_arr
                arr_match_score[idx] = mapali.arrangement_match_score(arrange_src, arrange_dst, tf)
                if print_messages: print ('match_score {:d}/{:d}: {:.4f}'.format(idx+1,tforms.shape[0], arr_match_score[idx]))

                best_idx = max(arr_match_score, key=arr_match_score.get)

        hypothesis = tforms[best_idx]
        n_cluster = 0

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
        if print_messages: print ('\t clustering {:d} transformations'.format(parameters.shape[0]))
        cls = sklearn.cluster.DBSCAN(eps=config['dbscan_eps'], min_samples=config['dbscan_min_samples'])
        cls.fit(parameters)
        labels = cls.labels_
        unique_labels = np.unique(labels)
        if print_messages: print ( '\t *** total: {:d} clusters...'.format(unique_labels.shape[0]-1) )

        ###  match_score for each cluster
        
        if config['multiprocessing']:
            cluster_representative = {}
            for lbl in np.setdiff1d(unique_labels,[-1]):
                class_member_idx = np.nonzero(labels == lbl)[0]
                class_member = [ tforms[idx] for idx in class_member_idx ]
                
                # pick the one that is closest to all others in the same group
                params = parameters[class_member_idx]
                dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(params, 'euclidean'))
                dist_arr = dist_mat.sum(axis=0)
                tf = class_member[ np.argmin(dist_arr) ]
                cluster_representative[lbl] = tf

            arrangement_match_score_par = partial( _arrangement_match_score_4partial,
                                                   arrangement_src=src_arr,
                                                   arrangement_dst=dst_arr,
                                                   tforms = cluster_representative)

            # note that keys to a dictionary (not guaranteed to be ordered ) are passes as idx
            # therefor, later, the index to max arg, is the index to keys of the dictionary
            with ctx.closing(mp.Pool(processes=4)) as p:
                arr_match_score = p.map( arrangement_match_score_par, cluster_representative.keys())

            max_idx = np.argmax(arr_match_score)
            winning_cluster_key = cluster_representative.keys()[max_idx]
            winning_cluster = [tforms[idx] for idx in np.nonzero(labels==winning_cluster_key)[0]]

        else:
            arr_match_score = {}
            for lbl in np.setdiff1d(unique_labels,[-1]):
                class_member_idx = np.nonzero(labels == lbl)[0]
                class_member = [ tforms[idx] for idx in class_member_idx ]
                
                # pick the one that is closest to all others in the same group
                params = parameters[class_member_idx]
                dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(params, 'euclidean'))
                dist_arr = dist_mat.sum(axis=0)
                tf = class_member[ np.argmin(dist_arr) ]
                
                arrange_src = src_arr
                arrange_dst = dst_arr
                arr_match_score[lbl] = mapali.arrangement_match_score(arrange_src, arrange_dst, tf)
                if print_messages: print ('match score cluster {:d}/{:d}: {:.4f}'.format(lbl,len(unique_labels)-1, arr_match_score[lbl]) )

                ### pick the winning cluster
                winning_cluster_idx = max(arr_match_score, key=arr_match_score.get)
                winning_cluster = [tforms[idx] for idx in np.nonzero(labels==winning_cluster_idx)[0]]

        ### match_score for the entities of the winning cluster
        ### there will be too few to cluster here
        arr_match_score = {}
        for idx, tf in enumerate(winning_cluster):
            arrange_src = src_arr
            arrange_dst = dst_arr
            arr_match_score[idx] = mapali.arrangement_match_score(arrange_src, arrange_dst, tf)
            if print_messages: print ('match score element {:d}/{:d}: {:.4f}'.format(idx,len(winning_cluster)-1, arr_match_score[idx]) )

        ### pick the wining cluster
        hypothesis_idx = max(arr_match_score, key=arr_match_score.get)
        hypothesis =  winning_cluster[hypothesis_idx]
        
        n_cluster = unique_labels.shape[0]-1

    # np.save('arr_match_score_'+'_'.join(keys)+'.npy', arr_match_score)
    elapsed_time = time.time() - tic


    return hypothesis, n_cluster, elapsed_time


################################################################################
def _visualize_save(src_results, dst_results, hypothesis,
                    visualize=True, save_to_file=False, details=None):
    '''
    '''

    fig, axes = plt.subplots(1,3, figsize=(20,12))

    # src
    axes[0].imshow(src_results['image'], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
    # axes[0].imshow(src_results['distance'], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
    mapali.maplt.plot_arrangement(axes[0], src_results['arrangement'], printLabels=False)

    # dst
    axes[1].imshow(dst_results['image'], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
    # axes[1].imshow(dst_results['distance'], cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
    mapali.maplt.plot_arrangement(axes[1], dst_results['arrangement'], printLabels=False)

    # result
    aff2d = matplotlib.transforms.Affine2D( hypothesis.params )
    im_dst = axes[2].imshow(dst_results['image'], origin='lower', cmap='gray', alpha=.5, clip_on=True)
    im_src = axes[2].imshow(src_results['image'], origin='lower', cmap='gray', alpha=.5, clip_on=True)
    im_src.set_transform( aff2d + axes[2].transData )
    # finding the extent of of dst and transformed src
    xmin_d,xmax_d, ymin_d,ymax_d = im_dst.get_extent()
    x1, x2, y1, y2 = im_src.get_extent()
    pts = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    pts_tfrom = aff2d.transform(pts)    
    xmin_s, xmax_s = np.min(pts_tfrom[:,0]), np.max(pts_tfrom[:,0]) 
    ymin_s, ymax_s = np.min(pts_tfrom[:,1]), np.max(pts_tfrom[:,1])
    axes[2].set_xlim( min(xmin_s,xmin_d), max(xmax_s,xmax_d) )
    axes[2].set_ylim( min(ymin_s,ymin_d), max(ymax_s,ymax_d) )

    if visualize and save_to_file:
        np.save(save_to_file+'_details.npy', details)
        plt.savefig(save_to_file+'.png', bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    elif visualize:
        plt.tight_layout()
        plt.show()

    if save_to_file:
        np.save(save_to_file+'_details.npy', details)
        plt.savefig(save_to_file+'.png', bbox_inches='tight')
        plt.close(fig)

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
def _arrangement_match_score_4partial(idx, arrangement_src, arrangement_dst, tforms):
    return mapali.arrangement_match_score(arrangement_src, arrangement_dst, tforms[idx])


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

    example
    -------
    python -i align_maps_script.py --img_src '/home/saesha/Documents/tango/HIH_01_full/20170131135829.png' --img_dst '/home/saesha/Dropbox/myGits/sample_data/HH/HIH/HIH_04.png' -multiprocessing -visualize -save_to_file

    python -i align_maps_script.py --img_src '/home/saesha/Documents/tango/E5_11/20170409125554.png' --img_dst '/home/saesha/Dropbox/myGits/sample_data/HH/E5/E5_06.png' -multiprocessing -visualize
    
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
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break   

    ### setting defaults values for undefined parameters 
        
    ### setting defaults values for visualization and saving options
    visualize = True if 'visualize' in options else False
    save_to_file = True if 'save_to_file' in options else False

    out_file_name = _extract_target_file_name(img_src, img_dst)
    save_to_file = out_file_name if save_to_file==True else False
    multiprocessing = True if 'multiprocessing' in options else False



    ################################################################################
    '''
    NOTE on binarization and occupancy threshold; only for trait detection, the
    theshold is set to 100, so unexplored is considered open space. For SKIZ,
    distance transform, and face occupancy ratio, unexplored are considered as 
    occupied and threshold is set to 200.
    '''
    print_messages = False
    
    ########## image loading, SKIZ, distance transform and trait detection
    lnl_config = {'binary_threshold_1': 200, # with numpy - for SKIZ and distance
                  'binary_threshold_2': [100, 255], # with cv2 - for trait detection
                  'traits_from_file': False, # else provide yaml file name
                  'trait_detection_source': 'binary_inverted',
                  'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size
                  'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
                  'orthogonal_orientations': True} # for dominant orientation detection

    src_results, src_lnl_t = _lock_n_load(img_src, lnl_config)
    dst_results, dst_lnl_t = _lock_n_load(img_dst, lnl_config)
    

    ########## arrangement (and pruning)
    arr_config = {'multi_processing':4, 'end_point':False, 'timing':False,
                  'prune_dis_neighborhood': 2,
                  'prune_dis_threshold': .075, # home:0.15 - office:0.075
                  'occupancy_threshold': 200} # cell below this is considered occupied
    
    src_results['arrangement'], src_arr_t = _construct_arrangement(src_results, arr_config)
    dst_results['arrangement'], dst_arr_t = _construct_arrangement(dst_results, arr_config)
    
    interpret_t = src_lnl_t + src_arr_t + dst_lnl_t + dst_arr_t 
    
    
    ########## Hypothesis generation
    hyp_config = { 'scale_mismatch_ratio_threshold': .3, # .5,
                   'scale_bounds': [.5, 2], #[.1, 10]
                   'face_occupancy_threshold': .5}
    
    tforms, hyp_gen_t, tforms_total, tforms_after_reject = _generate_hypothese(src_results['arrangement'],
                                                                               src_results['image'].shape,
                                                                               dst_results['arrangement'],
                                                                               dst_results['image'].shape,
                                                                               hyp_config)

    ########## pick the winning hypothesis
    sel_config = {'multiprocessing': multiprocessing,
                  'too_many_tforms': 3000,
                  'dbscan_eps': 0.051,
                  'dbscan_min_samples': 2}
    hypothesis, n_cluster, sel_win_t = _select_winning_hypothesis(src_results['arrangement'],
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

    _visualize_save(src_results, dst_results, hypothesis, visualize, save_to_file, details)

    time_key = ['src_lnl_t', 'dst_lnl_t', 'src_arr_t', 'dst_arr_t', 'hyp_gen_t']
    print ('total time: {:.5f}'.format( np.array([details[key] for key in time_key]).sum() ) )





################################################################################
####################################################################### dumpster
################################################################################

# print_messages = False
    
# ########## image loading, SKIZ, distance transform and trait detection
# lnl_config = {'binary_threshold_1': 200, # with numpy - for SKIZ and distance
#               'binary_threshold_2': [100, 255], # with cv2 - for trait detection
#               'traits_from_file': False, # else provide yaml file name
#               'trait_detection_source': 'binary_inverted',
#               'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size
#               'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
#               'orthogonal_orientations': True} # for dominant orientation detection


# file_name = '/home/saesha/Documents/tango/HIH_03/20170409123544.png'
# file_name = '/home/saesha/Documents/tango/kpt4a_kl/20170131162628.png'
# file_name = '/home/saesha/Dropbox/myGits/saeedghsh_tango_n_layout_maps/E5/E5_7.png'
# img = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )

# if 1:
#     img_ = np.flipud(img)
    
#     ### smoothing
#     img_ = cv2.blur(img_, (11,11))
#     img_ = cv2.GaussianBlur(img_, (11,11),0)    
    
#     ### oriented gradient of the image
#     # is it (dx - 1j*dy) or (dx + 1j*dy)
#     # this is related to "flipud" problem mentioned above
#     dx = cv2.Sobel(img_, cv2.CV_64F, 1,0, ksize=11)
#     dy = cv2.Sobel(img_, cv2.CV_64F, 0,1, ksize=11)
#     grd = dx - 1j*dy
#     plt.imshow(np.abs(grd), cmap='gray')
#     plt.show()
        
#     h,w = grd.shape[:2]
#     Ang_arr = np.reshape(np.angle(grd), h*w, 1)
#     Mag_arr = np.reshape(np.abs(grd), h*w, 1)
#     hist, bins  = np.histogram(Ang_arr, bins=180*5, weights=Mag_arr, density=True)
#     binc = (bins[:-1]+bins[1:])/2
    
#     wl = 21
#     his_sm = utils.smooth(hist, window_len=wl)
#     plt.plot(binc,hist, 'b')
#     plt.plot(binc,his_sm, 'r')
#     plt.show()




# if 1:
#     orientations = _find_dominant_orientations(img, orthogonal_orientations=True)
#     [thr1, thr2] = [100,255]
#     ret, bin_img = cv2.threshold(img , thr1,thr2 , cv2.THRESH_BINARY_INV)
#     image = bin_img

#     # 'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal]
#     traits = _find_lines_with_radiography(image, orientations, lnl_config['peak_detect_sinogram_config'])    
#     boundary = [-20, -20, image.shape[1]+20, image.shape[0]+20]
#     traits = mapali.arr.utls.unbound_traits(traits)
#     traits = mapali.arr.utls.bound_traits(traits, boundary)
#     arr_config = {'multi_processing':4, 'end_point':False, 'timing':False,
#                   'prune_dis_neighborhood': 2,
#                   'prune_dis_threshold': .075, # home:0.15 - office:0.075
#                   'occupancy_threshold': 200} # cell below this is considered occupied
    
#     arrange = mapali.arr.Arrangement(traits, arr_config)

#     fig, axes = plt.subplots(1,1, figsize=(12,12))
#     axes.imshow(img, cmap='gray', alpha=.7, interpolation='nearest', origin='lower')
#     mapali.maplt.plot_arrangement(axes, arrange, printLabels=False)
#     plt.show()


# image_ = np.flipud(image)
# imgcenter = (image.shape[1]/2, # cols == x
#              image.shape[0]/2) # rows == y

# sinog_angles = orientations - np.pi/2 # in radian
# sinograms = skimage.transform.radon(image_, theta=sinog_angles*180/np.pi )#, circle=True)
# sinogram_center = len(sinograms.T[0])/2


# peakind_my = utils.FindPeaks(sinograms.T[0] ,
#                              CWT=False, cwt_range=(1,8,1),
#                              Refine_win = int(15),
#                              MinPeakDist = int(15),
#                              MinPeakVal = 0.15,
#                              Polar=False)


# peakind_mys = utils.FindPeaks(utils.smooth(sinograms.T[0], window_len=11),
#                               CWT=False, cwt_range=(1,8,1),
#                               Refine_win = int(15),
#                               MinPeakDist = int(15),
#                               MinPeakVal = 0.15,
#                               Polar=False)

# # peakind_cwt = utils.FindPeaks(sinograms.T[0] ,
# #                               CWT=True, cwt_range=(10,150,5),
# #                               Refine_win = int(1),
# #                               MinPeakDist = int(1),
# #                               MinPeakVal = 0,
# #                               Polar=False)
# # print(len(peakind_cwt))


# plt.plot( peakind_mys, sinograms.T[0][np.array(peakind_mys)], 'ro' )
# plt.plot( peakind_my, sinograms.T[0][np.array(peakind_my)], 'g*' )

# plt.plot( sinograms.T[0], 'b' )
# # plt.plot( utils.smooth(sinograms.T[0], window_len=7), 'r' )
# plt.show()
