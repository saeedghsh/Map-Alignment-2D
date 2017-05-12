from __future__ import print_function


# import sys
# if sys.version_info[0] == 3:
#     from importlib import reload
# elif sys.version_info[0] == 2:
#     pass

# new_paths = [
#     u'../arrangement/', # this would be relative address from the runMe.y script that loads this module 
# ]
# for path in new_paths:
#     if not( path in sys.path):
#         sys.path.append( path )

import copy
import itertools 
# import collections

import cv2
import numpy as np
import numpy.linalg
import scipy
import scipy.ndimage
import sympy as sym
import networkx as nx
import sklearn.cluster
import skimage.transform

import matplotlib.transforms
import matplotlib.path as mpath

import Polygon#, Polygon.IO


import arrangement.arrangement as arr
# from . import mapali_plotting as maplt
import mapali_plotting as maplt # this is used in the runMe.py


################################################################################
dir_layout = '/home/saesha/Dropbox/myGits/sample_data/'
dir_tango = '/home/saesha/Documents/tango/'

data_sets = {
    # layouts
    'HIH_layout':     dir_layout+'HH/HIH/HIH_04.png',
    'E5_layout':      dir_layout+'HH/E5/E5_06.png',
    'F5_layout':      dir_layout+'HH/F5/F5_04.png',
    'kpt4a_layout':   dir_layout+'sweet_home/kpt4a.png',

    # tango maps
    'HIH_01_tango':   dir_tango+'HIH_01_full/20170131135829.png',
    'HIH_02_tango':   dir_tango+'HIH_02/20170409123351.png',
    'HIH_03_tango':   dir_tango+'HIH_03/20170409123544.png',
    'HIH_04_tango':   dir_tango+'HIH_04/20170409123754.png',

    'kpt4a_f_tango':  dir_tango+'kpt4a_f/20170131163311.png',
    'kpt4a_kb_tango': dir_tango+'kpt4a_kb/20170131163634.png',
    'kpt4a_kl_tango': dir_tango+'kpt4a_kl/20170131162628.png',
    'kpt4a_lb_tango': dir_tango+'kpt4a_lb/20170131164048.png',

    'E5_01_tango':    dir_tango+'E5_1/20170131150415.png',
    'E5_02_tango':    dir_tango+'E5_2/20170131131405.png',
    'E5_03_tango':    dir_tango+'E5_3/20170131130616.png',
    'E5_04_tango':    dir_tango+'E5_4/20170131122040.png',
    'E5_05_tango':    dir_tango+'E5_5/20170205104625.png',  
    'E5_06_tango':    dir_tango+'E5_6/20170205105917.png',
    'E5_07_tango':    dir_tango+'E5_7/20170205111301.png',
    'E5_08_tango':    dir_tango+'E5_8/20170205112339.png',
    'E5_09_tango':    dir_tango+'E5_9/20170205110552.png',
    'E5_10_tango':    dir_tango+'E5_10/20170205111807.png',
    'E5_11_tango':    dir_tango+'E5_11/20170409125554.png',
    'E5_12_tango':    dir_tango+'E5_12/20170409130127.png',
    'E5_13_tango':    dir_tango+'E5_13/20170409130542.png',
    'E5_14_tango':    dir_tango+'E5_14/20170409131152.png',
    
    'F5_01_tango':    dir_tango+'F5_1/20170131132256.png',
    'F5_02_tango':    dir_tango+'F5_2/20170131125250.png',
    'F5_03_tango':    dir_tango+'F5_3/20170205114543.png',
    'F5_04_tango':    dir_tango+'F5_4/20170205115252.png',
    'F5_05_tango':    dir_tango+'F5_5/20170205115820.png',
    'F5_06_tango':    dir_tango+'F5_6/20170205114156.png',
    'F5_07_tango':    dir_tango+'F5_7/20170409113201.png',
    'F5_08_tango':    dir_tango+'F5_8/20170409113636.png',
    'F5_09_tango':    dir_tango+'F5_9/20170409114748.png',
    'F5_10_tango':    dir_tango+'F5_10/20170409115054.png',
    'F5_11_tango':    dir_tango+'F5_11/20170409115625.png',
    'F5_12_tango':    dir_tango+'F5_12/20170409120348.png',
    'F5_13_tango':    dir_tango+'F5_13/20170409120957.png',
    'F5_14_tango':    dir_tango+'F5_14/20170409121712.png',

}

################################################################################
def reject_implausible_transformations(transformations,
                                       image_src_shape, image_dst_shape,
                                       scale_mismatch_ratio_threshold=.1,
                                       scale_bounds=[.1, 10] ):
    '''
    Input
    -----
    transformations (numpy array)
    each element is a skimage.transform.AffineTransform object
    (affine transforms, ie. tf.scale is a tuple)

    image_src_shape, image_dst_shape
    (assuming images are vertically aligned with x-y axes)    

    Parameters
    ----------
    scale_mismatch_ratio_threshold (default: .1)
    scale_bounds (default: [.1, 10])
    '''

    ### rejecting transformations that contain nan
    NaN_free_idx = []
    for idx,tf in enumerate(transformations):
        if not np.any( np.isnan( tf.params )):
            NaN_free_idx.append(idx)
    # print ( 'nan_reject: {:d}'.format( len(transformations)-len(NaN_free_idx) ) )
    transformations = transformations[NaN_free_idx]

    ### reject transformations with mismatching scale or extrem scales
    correct_scale_idx = []
    for idx,tf in enumerate(transformations):
        if np.abs(tf.scale[0]-tf.scale[1])/np.min(tf.scale) < scale_mismatch_ratio_threshold:
            if (scale_bounds[0] < tf.scale[0] < scale_bounds[1]):
                correct_scale_idx.append(idx)
    # print ( 'scale_reject: {:d}'.format( len(transformations)-len(correct_scale_idx) ) )
    transformations = transformations[correct_scale_idx]

    ### reject transformations, if images won't overlap under the transformation
    src_h, src_w = image_src_shape
    src = np.array([ [0,0], [src_w,0], [src_w,src_h], [0,src_h], [0,0] ])
    
    dst_h, dst_w = image_dst_shape
    dst = np.array([ [0,0], [dst_w,0], [dst_w,dst_h], [0,dst_h], [0,0] ])
    dst_path = create_mpath(dst)
    
    overlapping_idx = []
    for idx,tf in enumerate(transformations):
        src_warp = tf._apply_mat(src, tf.params)
        src_warp_path = create_mpath(src_warp)
        if src_warp_path.intersects_path(dst_path,filled=True):
            overlapping_idx.append(idx)
    # print ('non_overlapping_reject: {:d}'.format(len(transformations)-len(overlapping_idx) ) )

    transformations = transformations[overlapping_idx]    
    
    return transformations



################################################################################
def align_ombb(face_src,face_dst, tform_type='similarity'):
    '''
    move to arr.utls
    '''
    src = face_src.attributes['ombb_path'].vertices[:-1,:]
    dst = face_dst.attributes['ombb_path'].vertices[:-1,:]

    alignments = [ skimage.transform.estimate_transform( tform_type, np.roll(src,-roll,axis=0), dst )
                   for roll in range(4) ]
    return alignments 


################################################################################
def set_ombb_of_faces (arrangement):
    '''
    move to arr.utls
    '''
    for face in arrangement.decomposition.faces:
        ombb = oriented_minimum_bounding_box(face.path.vertices)
        face.attributes['ombb_path'] = create_mpath(ombb)
    return arrangement

################################################################################
def distance2point(p1,p2,p):
    '''
    move to arr.utls
    called by "oriented_minimum_bounding_box"
    
    (p1,p2) represents a line, not a segments
    input points are numpy arrays or lists
    '''
    (x0,y0), (x1,y1), (x2,y2) = p, p1, p2
    dx, dy = x2-x1, y2-y1
    return np.abs(dy*x0 -dx*y0 -x1*y2 +x2*y1) / np.sqrt(dx**2+dy**2)

################################################################################
def linesIntersectionPoint(P1,P2, P3,P4):
    '''
    move to arr.utls
    called by "oriented_minimum_bounding_box"
    
    line1 = P1,P2
    line2 = P3,P4
    
    returns the intersection "Point" of the two lines
    returns "None" if lines are parallel
    This function treats the line as infinit lines
    '''
    denom = (P1[0]-P2[0])*(P3[1]-P4[1]) - (P1[1]-P2[1])*(P3[0]-P4[0])
    if np.abs(denom) > np.spacing(1):
        num_x = ((P1[0]*P2[1])-(P1[1]*P2[0]))*(P3[0]-P4[0]) - (P1[0]-P2[0])*((P3[0]*P4[1])-(P3[1]*P4[0]))
        num_y = ((P1[0]*P2[1])-(P1[1]*P2[0]))*(P3[1]-P4[1]) - (P1[1]-P2[1])*((P3[0]*P4[1])-(P3[1]*P4[0]))
        return np.array([num_x/denom , num_y/denom])
    else:
        return None

################################################################################        
def convexHullArea(vertices):
    '''
    move to arr.utls

    called by "oriented_minimum_bounding_box"

    vertices passed to this function in the form of numpy array
    '''
    ### sorting vertices CCW
    center = np.mean(vertices,axis=0)
    angletoVertex = [np.arctan2(p[1]-center[1] , p[0]-center[0])
                     for p in vertices]
    v = np.array([v for (t,v) in sorted(zip(angletoVertex,vertices))])
    
    ### calculatig the area, as a sum of triangles
    area = [np.cross([v[i,0]-v[0,0],v[i,1]-v[0,1]] , [v[i+1,0]-v[0,0],v[i+1,1]-v[0,1]])
            for i in range(1,len(v)-1)]
    area = 0.5 * np.sum(np.abs(area))
    
    return area

################################################################################
def oriented_minimum_bounding_box (points):
    '''
    move to arr.utls

    arbitrary oriented minimum bounding box based on rotating caliper
    1_ for each simplex[i] in the polygon (segment between two consequtive vertices):
    a) line1 = simplex[i]
    b) pTemp = the fartherest vertex to line1
    c) line2 = parallel to line1, passing through pTemp
    d) caliper is the line1+line2 (two parallel line)
    e) lineTemp = perpendicular to the caliper's lines
    f) pTemp3 = the fartherest vertex to the lineTemp
    g) line3 = parallel to lineTemp, passing through pTemp3
    h) pTemp4 = the fartherest vertex to the line3
    i) line4 = parallel to line3, passing through pTemp4
    j) mbb according to simplex[i]: p1,p2,p3,p4 = intersection(line1,line2,line3,line4)
    k) calculate the area covered by (p1,p2,p3,p4)
    2_ among all the mbb created based on all simplices, pick the one with smallest area
    '''

    boundingBoxes = []
    area = []

    hull = scipy.spatial.ConvexHull(points)
    verticesIdx = list(set([vertex for simplex in hull.simplices for vertex in simplex ]))
    for simplex in hull.simplices:
        p1,p2 = hull.points[simplex]

        l12t = np.arctan2(p2[1]-p1[1] , p2[0]-p1[0])
        l34t = l12t+np.pi/2
        
        # line1 = p1,p2
        l1p1, l1p2 = p1, p2

        # line2 = construct the parallel from vIdx 
        dis = list(np.abs([distance2point(l1p1, l1p2, points[p]) for p in verticesIdx]))
        val, idx = max((val, idx) for (idx, val) in enumerate(dis))
        vIdx = verticesIdx[idx]
        l2p1, l2p2 = points[vIdx], points[vIdx]+np.array([np.cos(l12t),np.sin(l12t)])
        
        # lineTemp = arbitrary perpendicular line to line1 and line2.
        ltp1,ltp2 = p1, p1+np.array([np.cos(l34t),np.sin(l34t)])
        
        # line3 = parallel to lineTemp, passing through farthest vertex to lineTemp
        dis = list(np.abs([distance2point(ltp1,ltp2, points[p])  for p in verticesIdx]))
        val, idx = max((val, idx) for (idx, val) in enumerate(dis))
        vIdx = verticesIdx[idx]
        l3p1, l3p2 = points[vIdx], points[vIdx]+np.array([np.cos(l34t),np.sin(l34t)])
        
        # line4 = parallel to line3, passing through farthest vertex to line3
        dis = list(np.abs([distance2point(l3p1,l3p2, points[p])  for p in verticesIdx]))
        val, idx = max((val, idx) for (idx, val) in enumerate(dis))
        vIdx = verticesIdx[idx]
        l4p1, l4p2 = points[vIdx], points[vIdx]+np.array([np.cos(l34t),np.sin(l34t)])

        # BoundingBox = 4 points (intersections of line 1 to 4)
        vertices = np.array([ linesIntersectionPoint(l1p1,l1p2, l3p1,l3p2),
                              linesIntersectionPoint(l1p1,l1p2, l4p1,l4p2),
                              linesIntersectionPoint(l2p1,l2p2, l3p1,l3p2),
                              linesIntersectionPoint(l2p1,l2p2, l4p1,l4p2) ])

        # a proper convexhull algorithm should be able to sort these!
        # but my version of scipy does not have the vertices yet!
        # so I sort them by the angle
        center = np.mean(vertices,axis=0)
        angle2Vertex = [np.arctan2(p[1]-center[1] , p[0]-center[0]) for p in vertices]
        ### sorting bb according to tt
        BB = np.array([ v for (t,v) in sorted(zip(angle2Vertex,vertices)) ])
        boundingBoxes.append(BB)
        
        # find the area covered by the BoundingBox
        area.append(convexHullArea(BB))
                
    # return the BoundingBox with smallest area
    val, idx = min((val, idx) for (idx, val) in enumerate(area))
       
    return boundingBoxes[idx]


################################################################################
def find_face2face_association(faces_src, faces_dst, aff2d=None):
    '''
    problems:
    this does not result in a one to one assignment
    '''

    face_area_src = np.array([face.get_area() for face in faces_src])
    face_area_dst = np.array([face.get_area() for face in faces_dst])
    # cdist expects 2d arrays as input, so I just convert the 1d area value
    # to 2d vectors, all with one (or any aribitrary number)
    face_area_src_2d = np.stack((face_area_src, np.ones((face_area_src.shape))),axis=1)
    face_area_dst_2d = np.stack((face_area_dst, np.ones((face_area_dst.shape))),axis=1)
    f2f_distance = scipy.spatial.distance.cdist(face_area_src_2d,
                                                face_area_dst_2d,
                                                'euclidean')

    if aff2d is None:
        face_cen_src = np.array([face.attributes['centre'] for face in faces_src])
    else:
        face_cen_src = aff2d.transform(np.array([face.attributes['centre'] for face in faces_src]))

    face_cen_dst = np.array([face.attributes['centre'] for face in faces_dst])
        
    f2f_association = {}
    for src_idx in range(f2f_distance.shape[0]):
        # if the centre of faces in dst are not inside the current face of src
        # their distance are set to max, so that they become illegible
        # TODO: should I also check for f_dst.path.contains_point(face_cen_src)

        f_src = faces_src[src_idx]
        contained_in_src = f_src.path.contains_points(face_cen_dst)
        contained_in_dst = [f_dst.path.contains_point(face_cen_src[src_idx])
                            for f_dst in faces_dst]
        contained = np.logical_and( contained_in_src, contained_in_dst)
        if any(contained):
            maxDist = f2f_distance[src_idx,:].max()
            distances = np.where(contained,
                                 f2f_distance[src_idx,:],
                                 np.repeat(maxDist, contained.shape[0] ))
            dst_idx = np.argmin(distances)
            f2f_association[src_idx] = dst_idx
        else:
            # no face of destination has center inside the current face of src
            pass

    return f2f_association

################################################################################
def arrangement_match_score(arrangement_src, arrangement_dst,
                            tform,
                            label_associations=None):
    '''
    tform:  a transformation instance of skimage lib
    '''
    # construct a matplotlib transformation instance (for transformation of paths )
    aff2d = matplotlib.transforms.Affine2D( tform.params )

    ### making a deepcopy of each arrangements, so not to disturb original copy
    ### apparantely it is not needed and works alright without copy! 
    ### because I already deepcopy them before passing to this method!
    # arrange_src = copy.deepcopy(arrangement_src)
    # arrange_dst = copy.deepcopy(arrangement_dst)
    arrange_src = arrangement_src
    arrange_dst = arrangement_dst

    ### transforming paths of faces, and updating centre points 
    faces_src = arrange_src.decomposition.faces
    # faces_src = [ face for face in arrange_src.decomposition.faces if face.attributes['label_vote'] != -1]
    for face in faces_src:
        face.path = face.path.transformed(aff2d)
        face.attributes['centre'] = np.mean(face.path.vertices[:-1,:], axis=0)

    faces_dst = arrange_dst.decomposition.faces
    # faces_dst = [ face for face in arrange_dst.decomposition.faces if face.attributes['label_vote'] != -1]
    # for face in faces_dst: face.attributes['centre'] = np.mean(face.path.vertices[:-1,:], axis=0)
    
    # find face to face association
    f2f_association = find_face2face_association(faces_src, faces_dst)#, aff2d)

    # find face to face match score (of associated faces)
    f2f_match_score = {(f1_idx,f2f_association[f1_idx]): None
                       for f1_idx in f2f_association.keys()}
    for (f1_idx,f2_idx) in f2f_match_score.keys():
        score = face_match_score(faces_src[f1_idx], faces_dst[f2_idx])#, aff2d)
        f2f_match_score[(f1_idx,f2_idx)] = score

    # find the weights of pairs of associated faces to arrangement match score
    face_pair_weight = {}
    for (f1_idx,f2_idx) in f2f_match_score.keys():
        # f1_area = faces_src[f1_idx].get_area()
        # f2_area = faces_dst[f2_idx].get_area()
    
        # f1_w = float(f1_area) / float(arrange_src_area)
        # f2_w = float(f2_area) / float(arrange_dst_area)
        
        # face_pair_weight[(f1_idx,f2_idx)] = np.min([f1_w, f2_w])
        face_pair_weight[(f1_idx,f2_idx)] = np.min([faces_src[f1_idx].attributes['area_weight'],
                                                    faces_dst[f2_idx].attributes['area_weight']])

    # computing arrangement match score
    if label_associations is None:
        arr_score = np.sum([face_pair_weight[(f1_idx,f2_idx)]*f2f_match_score[(f1_idx,f2_idx)]
                            for (f1_idx,f2_idx) in f2f_match_score.keys()])

    elif label_associations is not None:
        # find the face category label similarity
        fpw = face_pair_weight
        fms = f2f_match_score

        fls = {} # face_label_similarity
        for (f1_idx,f2_idx) in f2f_match_score.keys():
            # fls[(f1_idx,f2_idx)] = face_category_distance(faces_src[f1_idx],
            #                                               faces_dst[f2_idx],
            #                                               label_associations=label_associations)
            l1 = faces_src[f1_idx].attributes['label_vote']
            l2 = faces_dst[f2_idx].attributes['label_vote']
            fls[(f1_idx,f2_idx)] = 1 if label_associations[l1]==l2 else 0

        arr_score = np.sum([ fls[(f1_idx,f2_idx)]*fpw[(f1_idx,f2_idx)]*fms[(f1_idx,f2_idx)]
                             for (f1_idx,f2_idx) in f2f_match_score.keys()])


    return arr_score


################################################################################
def face_match_score(face_src, face_dst, aff2d=None):
    '''
    NOTE
    ----
    This face_match_score idea is based on the assumption that the level of
    abstraction between the two maps are comparable.
    Or is it? The face_match_score is used for all variation of alignment
    (hypotheses) between the same two maps, so the dicrepency between levels of
    abstraction should affect the face_match_score of all hypotheses uniformly.
    '''
    if aff2d is None:
        p1 = Polygon.Polygon( [tuple(v) for v in face_src.path.vertices] )
    else:
        p1 = Polygon.Polygon( [tuple(v) for v in face_src.path.transformed(aff2d).vertices] )

    p2 = Polygon.Polygon( [tuple(v) for v in face_dst.path.vertices] )
    union = p1 | p2
    intersection = p1 & p2
    if union.area() == 0:
        # if one of the faces has the area equal to zero 
        return 0.

    overlap_ratio = intersection.area() / union.area()
    overlap_score = (np.exp(overlap_ratio) - 1) / (np.e-1)
    return overlap_score

    # # compute the area of intersection and union
    # # NOTE
    # # ----
    # # Union and intersection area are computed by pixelating the paths, Obviously
    # # this is an approximation.... 
    # pixels_in_f1 = {tuple(p) for p in get_pixels_in_mpath(face1.path)}
    # pixels_in_f2 = {tuple(p) for p in get_pixels_in_mpath(face2.path)}

    # union = len( pixels_in_f1.union(pixels_in_f2) )
    # intersection = len( pixels_in_f1.intersection(pixels_in_f2) )

    # if union == 0:
    #     # if one of the faces has the area equal to zero 
    #     return 0.

    # # computing overlap ratio and score
    # # ratio and score \in [0,1]
    # overlap_ratio = float(intersection) / float(union)
    # overlap_score = (np.exp(overlap_ratio) - 1) / (np.e-1)

    # return overlap_score


################################################################################
def objectivefun_image (X , *arg):
    '''
    X: the set of variables to be optimized
    
    src: source image (model - template)
    dst: destination image (static scene - target)
    '''
    tx, ty, s, t = X 
    tform = skimage.transform.AffineTransform(scale=(s,s),
                                              rotation=t,
                                              translation=(tx,ty))
    src_image, dst_image = arg[0], arg[1]
    mse, l2 = mse_norm(src_image, dst_image, tform)
    
    return mse


################################################################################
def mse_norm(src_image, dst_image, tform):
    '''

    since I use distance images as input, their median/gaussian blur is not far
    from their own valeus, otherwise for other images, it would be better to apply
    a blurring before computing the errors
    '''
    ###### constructing the paths of mbb for images
    # get the extent of source image
    minX, maxX = 1, src_image.shape[1]-1 #skiping boundary poitns
    minY, maxY = 1, src_image.shape[0]-1 #skiping boundary poitns
    # mbb-path of src before transform
    src_mbb_pts = np.array([[minX,minY],[maxX,minY],[maxX,maxY],[minX,maxY],[minX,minY]])
    src_mbb_path = create_mpath ( src_mbb_pts )    
    # mbb-path of src after transform
    src_warp_mbb_pts = tform._apply_mat(src_mbb_pts, tform.params)
    src_warp_mbb_path = create_mpath ( src_warp_mbb_pts )

    # creat a list of coordinates of all pixels in src, before and after transform
    X = np.arange(minX, maxX, 1)
    Y = np.arange(minY, maxY, 1)
    X, Y = np.meshgrid(X, Y)
    src_idx = np.vstack( (X.flatten(), Y.flatten()) ).T
    src_idx_warp = tform._apply_mat(src_idx, tform.params).astype(int)
    
    # get the extent of destination image
    minX, maxX = 1, dst_image.shape[1]-1 #skiping boundary poitns
    minY, maxY = 1, dst_image.shape[0]-1 #skiping boundary poitns
    # creat a list of coordinates of all pixels in dst
    X = np.arange(minX, maxX, 1)
    Y = np.arange(minY, maxY, 1)
    X, Y = np.meshgrid(X, Y)
    dst_idx = np.vstack( (X.flatten(), Y.flatten()) ).T
    # mbb-path of dst
    dst_mbb_pts = np.array([[minX,minY],[maxX,minY],[maxX,maxY],[minX,maxY],[minX,minY]])
    dst_mbb_path = create_mpath ( dst_mbb_pts )
    

    ###### find the area of intersection (overlap) and union
    # easiest way to compute the intersection area is to count the number pixels
    # from one image in the mbb-path of the other.    
    # since the pixels in src_idx_warp are transformed (change of scale), they
    # no longer represent the area, so I have to count the number of dst_idx
    # containted by the src_warp_mbb_path
    in_warped_src_mbb = src_warp_mbb_path.contains_points(dst_idx)
    intersect_area = in_warped_src_mbb.nonzero()[0].shape[0]
    src_warp_area = get_mpath_area(src_warp_mbb_path)
    dst_area = get_mpath_area(dst_mbb_path)
    union_area = src_warp_area + dst_area - intersect_area
    # overlap_ratio = float(intersect_area)/union_area # \in [0,1]
    # overlap_score = np.log2(overlap_ratio+1) # \in [0,1]
    # overlap_error = 1-overlap_score # \in [0,1]
    # if there is no overlap, return high error value
    if intersect_area==0: return 127 , union_area/2.# return average error
    # if intersect_area==0: return 127 , 127*union_area/2.# return average error


    ###### computing l2-norm and MSE 
    # find those pixels of src (after warp) that are inside mbb of dst
    in_dst_mbb = dst_mbb_path.contains_points(src_idx_warp)
    # src_idx = src_idx[in_dst_mbb]
    # src_idx_warp = src_idx_warp[in_dst_mbb]
    src_overlap = src_image[src_idx[in_dst_mbb,1],src_idx[in_dst_mbb,0]]
    dst_overlap = dst_image[src_idx_warp[in_dst_mbb,1], src_idx_warp[in_dst_mbb,0]]

    # compute l2-norm
    l2 =  (src_overlap - dst_overlap).astype(float)**2
    l2 = np.sum(l2)
    l2 = np.sqrt(l2)
    # compute MSE (l2 averaged over intersection area)
    mse = l2 / float(dst_overlap.shape[0])

    
    if 1: print(mse, l2)
    return  mse, l2  # * overlap_error

################################################################################
def get_mpath_area(path):
    '''
    move to arr.utls


    TODO:
    Isn't this based on the assumption that the path in convex?
    '''

    polygon = path.to_polygons()
    x = polygon[0][:,0]
    y = polygon[0][:,1]
    PolyArea = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return PolyArea


################################################################################
def create_mpath ( points ):
    '''
    move to arr.utls


    note: points must be in order
    - copied from mesh to OGM
    '''
    
    # start path
    verts = [ (points[0,0], points[0,1]) ]
    codes = [ mpath.Path.MOVETO ]

    # construct path - only lineto
    for point in points[1:,:]:
        verts += [ (point[0], point[1]) ]
        codes += [ mpath.Path.LINETO ]

    # close path
    verts += [ (points[0,0], points[0,1]) ]
    codes += [ mpath.Path.CLOSEPOLY ]

    # return path
    return mpath.Path(verts, codes)



################################################################################
def label_association(arrangements, connectivity_maps):
    '''
    Inputs
    ------
    arrangements (dictionary)
    the keys are the map names and the values are arrangement instances of the maps

    connectivity_maps (dictionary)
    the keys are the map names and the values are connectivity map (graph) instances of the maps
    

    Output
    ------
    association - (dictionary)
    the keys are the labels in the first map (the first key in the arrangements.keys()),
    and the values are the corresponding labels from the second map (the second key in the arrangements.keys())

    Note
    ----
    Assumption: number of categories are the same for both maps


    Note
    ----
    features is a dictionary, storing features of nodes in connectivity maps
    The keys of the features dictionary are: '{:s}_{:d}'.format(map_key, label)
    The labels are feteched from the labels of the corresponding faces in the arrangements
    So the features dictionary has kxl entries (k:number of maps, l:number of labels)

    For features[map_key_l], the value is a numpy.array of (nx2), where:
    n is the number nodes in the current map (map_key) with the same label (label==l)
    Features of nodes are their "degrees" and "load centralities".
    '''

    # assuming the number of cateogries are the same for both maps
    keys = arrangements.keys()
    f = arrangements[keys[0]].decomposition.faces[0]
    labels = [ int(k) for k in f.attributes['label_count'].keys() ]
    
    # assuming label -1 is universal, we'll set it at the end
    labels.pop(labels.index(-1))

    ### constructing the "features" dictionary    
    features = {}
    for key in keys:
        for lbl in labels:
            fs = [ (connectivity_maps[key].node[n_idx]['features'][0], # degree
                    connectivity_maps[key].node[n_idx]['features'][3]) # load centrality
                   for n_idx in connectivity_maps[key].node.keys()
                   if arrangements[key].decomposition.faces[n_idx].attributes['label_vote'] == lbl]
            features['{:s}_{:d}'.format(key,lbl)] = np.array(fs)

    ### feature scaling (standardization)
    for key in keys:
        # std and mean of all features in current map (regardless of their place category labels) 
        # TD: should I standardize wrt (mean,std) of all maps?
        all_features = np.concatenate( [ features['{:s}_{:d}'.format(key,lbl)]
                                         for lbl in labels ] )
        std = np.std( all_features, axis=0 )
        mean = np.mean( all_features, axis=0 )
        
        # standardizing all features
        for lbl in labels:
            features['{:s}_{:d}'.format(key,lbl)] -= mean
            features['{:s}_{:d}'.format(key,lbl)] /= std


    # ####################
    # #################### mode1 - no gaurantee for one to one association
    # ####################
    # # assuming label -1 is universal, we'll set it as default
    # associations = {-1:-1}
    # # finding associations between labels by minimum distance between their
    # # corresponding sets of feature
    # for lbl1 in labels:
    #     S1 = np.cov(  features['{:s}_{:d}'.format(keys[0], lbl1)], rowvar=False )
    #     U1 = np.mean( features['{:s}_{:d}'.format(keys[0], lbl1)], axis=0 )
    #     dist = [ bhattacharyya_distance (S1, U1,
    #                                      S2 = np.cov(  features['{:s}_{:d}'.format(keys[1], lbl2)], rowvar=False ),
    #                                      U2 = np.mean( features['{:s}_{:d}'.format(keys[1], lbl2)], axis=0 ) )
    #              for lbl2 in labels ]
    #     idx = dist.index(min(dist))
    #     associations[lbl1] = labels[idx]


    ####################
    #################### mode2 - gauranteed one to one association
    #################### 
    # row indices (lbl1) are labels of the source map
    # col indices (lbl2) are labels of the destination map
    dist = np.array([ [ bhattacharyya_distance(
        S1 = np.cov( features['{:s}_{:d}'.format(keys[0], lbl1)], rowvar=False),
        U1 = np.mean( features['{:s}_{:d}'.format(keys[0], lbl1)], axis=0),
        S2 = np.cov( features['{:s}_{:d}'.format(keys[1], lbl2)], rowvar=False),
        U2 = np.mean( features['{:s}_{:d}'.format(keys[1], lbl2)], axis=0)
    )[0,0]
                        for lbl1 in labels]
                      for lbl2 in labels ])
    
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist)
    associations = {lbl1:lbl2 for lbl1,lbl2 in zip(row_ind,col_ind)}
    # assuming label -1 is universal, we'll set it as default
    associations[-1] = -1


    return associations


################################################################################
def bhattacharyya_distance (S1,U1, S2,U2):
    '''
    S: covariance matrix
    U: mean vector

    http://en.wikipedia.org/wiki/Bhattacharyya_distance
    '''

    # sometimes there is only one sample in the feature vector
    # and the resulting covariance is a single number (i.e invalide)
    if S1.shape!=(2,2) or np.linalg.det(S1)<np.spacing(10): S1 = np.eye(2)
    if S2.shape!=(2,2) or np.linalg.det(S2)<np.spacing(10): S2 = np.eye(2)

    S = (S1+S2) /2.0

    U1 = np.atleast_2d(U1)
    U2 = np.atleast_2d(U2)

    if U1.shape[0] > U1.shape[1]: # U1, U2 are (nx1)
        A = (1.0/8) *np.dot( (U1-U2).T, np.dot( np.linalg.inv(S), (U1-U2)) )
    else: #  # U1, U2  are (1xn)
        A = (1.0/8) *np.dot( (U1-U2), np.dot( np.linalg.inv(S), (U1-U2).T) )

    B = (1.0/2) *np.log( np.linalg.det(S) /np.sqrt(np.linalg.det(S1)*np.linalg.det(S2)) )

    return A+B


################################################################################
def profile_nodes(graph):
    '''
    Note:
    nx.eigenvector_centrality - Not defined for multigraphs
    nx.katz_centralit - not implemented for multigraph
    nx.katz_centrality_numpy - not implemented for multigraph
    nx.current_flow_closeness_centrality - only for connected graphs
    nx.edge_betweenness_centrality - for edges
    '''

    # L = nx.laplacian_matrix(connectivity_maps[key])
    L = nx.normalized_laplacian_matrix(graph)
    eigenvalues = numpy.linalg.eigvals(L.A)
    
    eigenvector_centrality = nx.eigenvector_centrality_numpy( graph )
    load_centrality = nx.load_centrality( graph)
    harmonic_centrality = nx.harmonic_centrality( graph )
    degree_centrality = nx.degree_centrality( graph )
    closeness_centrality = nx.closeness_centrality( graph )
    betweenness_centrality = nx.betweenness_centrality( graph )
    

    for idx, key in enumerate( graph.node.keys() ):
        graph.node[key]['features'] = [
            graph.degree()[key],         # node degree
            eigenvalues[idx],            # node eigenvalue
            eigenvector_centrality[key], # node eigenvector centrality
            load_centrality[key],        # node load centrality
            harmonic_centrality[key],    # node harmonic centrality
            degree_centrality[key],      # node degree centrality
            closeness_centrality[key],   # node closeness centrality
            betweenness_centrality[key]  # node betweenness centrality
        ]

    return graph

################################################################################
def get_pixels_in_mpath(path, image_shape=None):
    '''
    given a path and image_size, this method return all the pixels in the path
    that are inside the image
    '''

    # find the extent of the minimum bounding box, sunjected to image boundaries
    mbb = path.get_extents()
    if image_shape is not None:
        xmin = int( np.max ([mbb.xmin, 0]) )
        xmax = int( np.min ([mbb.xmax, image_shape[1]-1]) )
        ymin = int( np.max ([mbb.ymin, 0]) )
        ymax = int( np.min ([mbb.ymax, image_shape[0]-1]) )
    else:
        xmin = int( mbb.xmin )
        xmax = int( mbb.xmax )
        ymin = int( mbb.ymin )
        ymax = int( mbb.ymax )

    x, y = np.meshgrid( range(xmin,xmax+1), range(ymin,ymax+1) )
    mbb_pixels = np.stack( (x.flatten().T,y.flatten().T), axis=1)
    
    in_path = path.contains_points(mbb_pixels)

    return mbb_pixels[in_path, :]

################################################################################
def assign_label_to_face(label_image, face, all_pixels=None):
    '''
    '''
    if all_pixels is None:
        x, y = np.meshgrid( np.arange(label_image.shape[1]),
                            np.arange(label_image.shape[0]))
        all_pixels = np.stack( (x.flatten(), y.flatten() ), axis=1)
        
        
    in_face = face.path.contains_points(all_pixels)
    pixels = all_pixels[in_face, :]

    if pixels.shape[0]==0:
        label = -1
        labels = { lbl: 0. for lbl in np.unique(label_image) }

    else:
        # mode=='vote'
        not_nan = np.nonzero( np.isnan(label_image[pixels[:,1],pixels[:,0]])==False )[0]
        label = np.median(label_image[pixels[:,1],pixels[:,0]][not_nan] )
        label = -1 if np.isnan(label) else label
        if label != int(label): #raise(NameError('same number of labels - median is conf.'))
            label = int(label) #
            print('here is a face which confuses the median...')


        # mode=='count'
        total = float(pixels.shape[0])
        labels = { lbl: np.nonzero(label_image[pixels[:,1],pixels[:,0]]==lbl)[0].shape[0] /total
                   for lbl in np.unique(label_image)}
        # assert np.abs( np.sum([ labels[lbl] for lbl in labels.keys() ]) -1) < np.spacing(10**5)

    face.attributes['label_vote'] = label
    face.attributes['label_count'] = labels
    # return face


################################################################################
def assign_label_to_all_faces(arrangement, label_image):
    '''
    attributes['label_vote'] (int)
    winner takes all. this contains a single value, that is the most common label in the face

    attributes['label_count'] (dictionary)
    per label in the label_image, there is a key in this dictionary
    the value to each key represents the presence of that label in the face (in percent [0,1]) 
    '''
    # note that all_pixels is in (col,row) format
    # use the same for "path.contains_points" and convert to (row,col) for
    # indexing the label_image
    x, y = np.meshgrid( np.arange(label_image.shape[1]),
                        np.arange(label_image.shape[0]))
    all_pixels = np.stack( (x.flatten(), y.flatten() ), axis=1)
    
    for idx, face in enumerate(arrangement.decomposition.faces):
        # set face attributes ['label_vote'], ['label_count']
        assign_label_to_face(label_image, face, all_pixels=all_pixels)
        # can't set the following since faces is a tuple, and no need to
        # arrangement.decomposition.faces[idx] = face

    return arrangement    

################################################################################
def face_category_distance(face1,face2, label_associations=None):
    '''

    label_associations
    keys are the place category labels in face1
    values corresponding to each key are the place category labels in face2
    ie. lbl1 corresponds to lb2 <=> label_associations[lbl1]=lbl2
    if label_associations is None, a direct correspondance is assumed

    Note
    ----
    if label_associations is provided, it is assumed that the its keys correspond
    to face1.attributes['label_count'].keys() and the values in the
    label_associations correspond to face2.attributes['label_count'].keys()

    Note
    ----
    it is assumed that the number of place category labels in the two faces are
    the same;
    ie. len(face1.attributes['label_count']) == len(face2.attributes['label_count'])    
    '''

    # since the difference between lables in face1 and face2 might be non-empty
    # the sequence of if-elif will consider unique labels in each face
    # otherwise they could be set as np.array and compute distance faster.
    # w1 = face1.attributes['label_count']
    # w2 = face2.attributes['label_count']    
    # dis = 0		
    # for lbl in set( w1.keys()+w2.keys() ):
    #     if (lbl in w1) and (lbl in w2):
    #         dis += (w1[lbl]-w2[lbl])**2
    #     elif lbl in w1:
    #         dis += w1[lbl]**2
    #     elif lbl in w2:
    #         dis += w2[lbl]**2            
    # dis = np.sqrt( dis )

    w1 = face1.attributes['label_count']
    w2 = face2.attributes['label_count']

    if label_associations is None:
        # assuming a direct correspondance between w1.keys() and w2.keys()
        label_associations = {key:key for key in w1.keys()}

    w1_arr = np.array([ w1[key]
                        for key in label_associations.keys() ])
    w2_arr = np.array([ w2[label_associations[key]]
                        for key in label_associations.keys() ])

    dis = np.sqrt( np.sum( (w1_arr-w2_arr)**2 ) )

    return dis

################################################################################
def are_same_category(face1,face2, label_associations=None, thr=.4):
    '''
    This method checks if the two input faces are similar according to 
    their place category label (count version)
    for the detials on the "count" version see: assign_label_to_face.__doc__

    Inputs
    ------
    face1, face2 ( Face instances )

    Parameters
    ----------
    label_associations (dictionary, default None)
    if the two faces belong to two different arrangments, there is no gaurantee
    that their labels correctly correspond to each other.
    to get label_associations, call the method "label_association()".
    If not provided (default None), it is assumed the two faces belong to the 
    same arrangement and there for the correspondance are direct.

    thr (float between (0,1), default: 0.4)
    If the distance between the category of faces is below this, the faces are
    assumed to belong to the same category

    Note
    ----
    It is required that the "assign_label_to_face()" method is called
    before calling this method.
    '''
    dis = face_category_distance( face1,face2, label_associations )    
    return True if dis<thr else False

################################################################################
def set_face_centre_attribute(arrangement,
                              source=['nodes','path'][0]):
    '''
    assumes all the faces in arrangement are convex
    
    if source == 'nodes' -> centre from nodes of arrangement
    if source == 'path' -> centre from vertrices of the face
    (for source == 'path', path must be up-todate)
    '''
    for face in arrangement.decomposition.faces:

        if source == 'nodes':
            nodes = [arrangement.graph.node[fn_idx]
                     for fn_idx in face.get_all_nodes_Idx()]
            xc = np.mean([ node['obj'].point.x for node in nodes ])
            yc = np.mean([ node['obj'].point.y for node in nodes ])
            face.attributes['centre'] = [float(xc),float(yc)]
        elif source == 'path':
            face.attributes['centre'] = np.mean(face.path.vertices[:-1,:], axis=0)

################################################################################
def construct_connectivity_map(arrangement, set_coordinates=True):
    '''

    Parameter:
    set_coordinates: Boolean (default:True)
    assumes all the faces in arrangement are convex, and set the coordinate of
    each corresponding node in the connectivity-map to the center of gravity of
    the face's nodes

    Note
    ----
    The keys to nodes in connectivity_map corresponds to 
    face indices in the arrangement
    '''

    connectivity_map = nx.MultiGraph()

    if set_coordinates: set_face_centre_attribute(arrangement)
    faces = arrangement.decomposition.faces

    ########## node construction (one node per each face)
    nodes = [ [f_idx, {}] for f_idx,face in enumerate(faces) ]
    connectivity_map.add_nodes_from( nodes )

    # assuming convex faces, node coordinate = COG (face.nodes)
    if set_coordinates:
        for f_idx,face in enumerate(faces):
            connectivity_map.node[f_idx]['coordinate'] = face.attributes['centre']

    ########## edge construction (add if faces are neighbor and connected)
    corssed_halfedges = [ (s,e,k)
                          for (s,e,k) in arrangement.graph.edges(keys=True)
                          if arrangement.graph[s][e][k]['obj'].attributes['crossed'] ]

    # todo: detecting topologically distict connection between face
    # consider this:
    # a square with a non-tangent circle enclosed and a vetical line in middle
    # the square.substract(circle) region is split to two and they are connected
    # through two topologically distict paths. hence, the graph of connectivity
    # map must be multi. But if two faces are connected with different pairs of 
    # half-edge that are adjacent, these connection pathes are not topologically
    # distict, hence they should be treated as one connection

    # todo: if the todo above is done, include it in dual_graph of the 
    # arrangement

    # for every pair of faces an edge is added if
    # faces are neighbours and the shared-half_edges are crossed 
    for (f1Idx,f2Idx) in itertools.combinations( range(len(faces) ), 2):
        mutualHalfEdges = arrangement.decomposition.find_mutual_halfEdges(f1Idx, f2Idx)
        mutualHalfEdges = list( set(mutualHalfEdges).intersection(set(corssed_halfedges)) )
        if len(mutualHalfEdges) > 0:
            connectivity_map.add_edges_from( [ (f1Idx,f2Idx, {}) ] )
            
    return arrangement, connectivity_map


################################################################################
def skiz_bitmap (image, invert=True, return_distance=False):
    '''
    Skeleton of Influence Zone [AKA; Generalized Voronoi Diagram (GVD)]

    Input
    -----
    Bitmap image (occupancy map)
    occupied regions: low value (black), open regions: high value (white)

    Parameter
    ---------
    invert: Boolean (default:False)
    If False, the ridges will be high (white) and backgroud will be low (black) 
    If True, the ridges will be low (black) and backgroud will be high (white) 

    Output
    ------
    Bitmap image (Skeleton of Influence Zone)


    to play with
    ------------
    > the threshold (.8) multiplied with grd_abs.max()
    grd_binary_inv = np.where( grd_abs < 0.8*grd_abs.max(), 1, 0 )
    > Morphology of the input image
    > Morphology of the grd_binary_inv
    > Morphology of the skiz
    '''

    original = image.copy()

    ###### image erosion to thicken the outline
    kernel = np.ones((9,9),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)
    # image = cv2.medianBlur(image, 5)
    # image = cv2.GaussianBlur(image, (5,5), 0).astype(np.uint8)
    # image = cv2.erode(image, kernel, iterations = 1) 
    image = cv2.medianBlur(image, 5)

    ###### compute distance image
    dis = scipy.ndimage.morphology.distance_transform_bf( image )
    # dis = scipy.ndimage.morphology.distance_transform_cdt(image)
    # dis = scipy.ndimage.morphology.distance_transform_edt(image)

    ###### compute gradient of the distance image
    dx = cv2.Sobel(dis, cv2.CV_64F, 1,0, ksize=5)
    dy = cv2.Sobel(dis, cv2.CV_64F, 0,1, ksize=5)
    grd = dx - 1j*dy
    grd_abs = np.abs(grd)

    # at some points on the skiz tree, the abs(grd) is very weak
    # this erosion fortifies those points 
    kernel = np.ones((3,3),np.uint8)
    grd_abs = cv2.erode(grd_abs, kernel, iterations = 1)

    # only places where gradient is low
    grd_binary_inv = np.where( grd_abs < 0.8*grd_abs.max(), 1, 0 )
    
    ###### skiz image
    # sometimes the grd_binary_inv becomes high value near the boundaries
    # the erosion of the image means to extend the low-level boundaries
    # and mask those undesired points
    kernel = np.ones((5,5),np.uint8)
    image = cv2.erode(image, kernel, iterations = 1)    
    skiz = (grd_binary_inv *image ).astype(np.uint8)

    # ###### map to [0,255]
    # skiz = (255 * skiz.astype(np.float)/skiz.max()).astype(np.uint8)

    ###### sometimes border lines are marked, I don't link it!
    # skiz[:,0] = 0
    # skiz[:,skiz.shape[1]-1] = 0
    # skiz[0,:] = 0
    # skiz[skiz.shape[0]-1,:] = 0


    # ###### post-processing
    kernel = np.ones((3,3),np.uint8)
    skiz = cv2.dilate(skiz, kernel, iterations = 1)
    skiz = cv2.erode(skiz, kernel, iterations = 1)
    skiz = cv2.medianBlur(skiz, 3)
    # kernel = np.ones((3,3),np.uint8)
    # skiz = cv2.dilate(skiz, kernel, iterations = 1)

    ###### inverting 
    if invert:
        thr1,thr2 = [127, 255]
        ret, skiz = cv2.threshold(skiz , thr1,thr2 , cv2.THRESH_BINARY_INV)
        

    if return_distance:
        return skiz, dis
    elif not(return_distance):
        return skiz


################################################################################
def set_edge_crossing_attribute(arrangement, skiz,
                                neighborhood=3, cross_thr=12):
    '''
    Parameters
    ----------
    neighborhood = 5 # the bigger, I think the more robust it is wrt skiz-noise
    cross_thr = 4 # seems a good guess! smaller also works
    
    Note
    ----
    skiz lines are usually about 3 pixel wide, so a proper edge crossing would
    result in about (2*neighborhood) * 3pixels ~ 6*neighborhood
    for safty, let's set it to 3*neighborhood

    for a an insight to its distributions:
    >>> plt.hist( [arrangement.graph[s][e][k]['obj'].attributes['skiz_crossing'][0]
                   for (s,e,k) in arrangement.graph.edges(keys=True)],
                  bins=30)
    >>> plt.show()

    Note
    ----
    since "set_edge_occupancy" counts low_values as occupied,
    (invert=True) must be set when calling the skiz_bitmap 
    '''

    set_edge_occupancy(arrangement,
                       skiz, occupancy_thr=127,
                       neighborhood=neighborhood,
                       attribute_key='skiz_crossing')


    for (s,e,k) in arrangement.graph.edges(keys=True):
        o, n = arrangement.graph[s][e][k]['obj'].attributes['skiz_crossing']
        arrangement.graph[s][e][k]['obj'].attributes['crossed'] = False if o <= cross_thr else True

    # since the outer part of arrangement is all one face (Null),
    # I'd rather not have any of the internal faces be connected with Null
    # that might mess up the structure/topology of the connectivity graph
    forbidden_edges  = arrangement.get_boundary_halfedges()
    for (s,e,k) in forbidden_edges:
        arrangement.graph[s][e][k]['obj'].attributes['crossed'] = False
        (ts,te,tk) = arrangement.graph[s][e][k]['obj'].twinIdx
        arrangement.graph[ts][te][tk]['obj'].attributes['crossed'] = False
    return skiz




################################################################################
def pixel_neighborhood_of_segment (p1,p2, neighborhood=5):
    '''
    move to arr.utls

    Input:
    ------
    p1 and p2, the ending points of a line segment

    Parameters:
    -----------
    neighborhood:
    half the window size in pixels (default:5)

    Output:
    -------
    neighbors: (nx2) np.array
    

    Note:
    Internal naming convention for the coordinates order is:
    (x,y) for coordinates - (col,row) for image
    However, this method is not concerned with the order.
    If p1 and p2 are passed as (y,x)/(row,col) the output will
    follow the convention in inpnut.
    '''
    # creating a uniform distribution of points on the line
    # for small segments N will be zero, so N = max(2,N)
    N = int( np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) )
    N = np.max([2, N])
    x = np.linspace(p1[0],p2[0], N, endpoint=True)
    y = np.linspace(p1[1],p2[1], N, endpoint=True)
    line_pts = np.stack( (x.T,y.T), axis=1)

    # index to all points in the minimum bounding box (MBB)
    # (MBB of the line from p1 to p2) + margin
    xMin = np.min([ int(p1[0]), int(p2[0]) ]) - neighborhood
    xMax = np.max([ int(p1[0]), int(p2[0]) ]) + neighborhood
    yMin = np.min([ int(p1[1]), int(p2[1]) ]) - neighborhood
    yMax = np.max([ int(p1[1]), int(p2[1]) ]) + neighborhood
    x, y = np.meshgrid( range(xMin,xMax), range(yMin,yMax) )
    mbb_pixels = np.stack( (x.flatten().T,y.flatten().T), axis=1)

    # min distance between points in MBB and the line
    dists = scipy.spatial.distance.cdist(line_pts, mbb_pixels, 'euclidean')
    dists = dists.min(axis=0)

    # flagging all the points that are in the neighborhood
    # of the 
    neighbors_idx = np.nonzero( dists<neighborhood )[0]
    neighbors = mbb_pixels[neighbors_idx]
    
    return neighbors

################################################################################
def pixel_neighborhood_of_halfedge (arrangement, (s,e,k),
                                    neighborhood=5, image_size=None):
    '''
    move to arr.arr
    
    Inputs:
    -------
    arrangement:
    (s,e,k): an index-set to a half-edge    

    Parameters:
    -----------
    neighborhood:
    half the window size in pixels (default:5)

    output:
    -------
    integer coordinate (index) to the half-edge's enighborhood.

    Note:
    -----
    output is in this format:
    (x,y) / (col,row)
    for platting use directly
    for image indexing use inverted
    '''

    he = arrangement.graph[s][e][k]['obj']
    trait = arrangement.traits[he.traitIdx]
    pt_1 = arrangement.graph.node[s]['obj'].point
    pt_2 = arrangement.graph.node[e]['obj'].point

    if not( isinstance(trait.obj, (sym.Line, sym.Segment, sym.Ray) ) ):
        raise (NameError(' only line trait are supported for now '))
    
    # Assuming only line segmnent - no arc-circle
    p1 = np.array([pt_1.x,pt_1.y]).astype(float)
    p2 = np.array([pt_2.x,pt_2.y]).astype(float)
    
    neighbors = pixel_neighborhood_of_segment (p1,p2, neighborhood)

    if image_size is None:
        return neighbors
    else:
        xMin, yMin, xMax, yMax = [0,0, image_size[1], image_size[0]]
        x_in_bound = (xMin<=neighbors[:,0]) & (neighbors[:,0]<xMax)
        y_in_bound = (yMin<=neighbors[:,1]) & (neighbors[:,1]<yMax)
        pt_in_bound = x_in_bound & y_in_bound
        in_bound_idx = np.where(pt_in_bound)[0]
        return neighbors[in_bound_idx]

################################################################################
def set_edge_distance_value(arrangement, distance_image, neighborhood):
    ''''''
    for (s,e,k) in arrangement.graph.edges(keys=True):
        neighbors = pixel_neighborhood_of_halfedge (arrangement, (s,e,k),
                                                    neighborhood=neighborhood,
                                                    image_size=distance_image.shape)

        neighbors_val = distance_image[neighbors[:,1], neighbors[:,0]]
        neighbors_val = neighbors_val[~np.isnan(neighbors_val)]
        arrangement.graph[s][e][k]['obj'].attributes['distances'] = neighbors_val

################################################################################
def set_edge_occupancy(arrangement,
                       image, occupancy_thr=200,
                       neighborhood=10,
                       attribute_key='occupancy'):
    '''
    This method sets the occupancy every edge in the arrangement wrt image:
    arrangement.graph[s][e][k]['obj'].attributes[attribute_key] = [occupied, neighborhood_area]

    Inputs
    ------
    arrangement:
    The arrangement corresponding to the input image

    image: bitmap (gray scale)
    The image represensts the occupancy map and has high value for open space.


    Parameters
    ----------
    occupancy_thr: default:200
    Any pixel with value below "occupancy_thr" is considered occupied.

    neighborhood: default=10
    half the window size (ie disk radius) that defines the neighnorhood

    attribute_key: default:'occupancy'
    The key to attribute dictionary of the edges to store the 'occupancy'
    This method is used for measuring occupancy of edges against occupancy map and skiz_map
    Therefor it is important to store the result in the atrribute dictionary with proper key


    Note
    ----
    "neighborhood_area" is dependant on the "neighborhood" parameter and the length of the edge.
    Hence it is the different from edge to edge.
    '''
    for (s,e,k) in arrangement.graph.edges(keys=True):
        neighbors = pixel_neighborhood_of_halfedge (arrangement, (s,e,k),
                                                    neighborhood,
                                                    image_size=image.shape)

        neighbors_val = image[neighbors[:,1], neighbors[:,0]]
        occupied = np.nonzero(neighbors_val<occupancy_thr)[0]
        
        # if neighbors.shape[0] is zero, I will face division by zero
        # when checking for occupancy ratio
        o = occupied.shape[0]
        n = np.max([1,neighbors.shape[0]])

        arrangement.graph[s][e][k]['obj'].attributes[attribute_key] = [o, n]


################################################################################
def prune_arrangement_with_distance(arrangement, distance_image,
                                    neighborhood=2, distance_threshold=.075):
    '''
    '''

    set_edge_distance_value(arrangement, distance_image, neighborhood=neighborhood)

    # get edges_to_prun
    forbidden_edges  = arrangement.get_boundary_halfedges()

    edges_to_purge = []
    for (s,e,k) in arrangement.graph.edges(keys=True):
        
        # rule 1: (self and twin) not in forbidden_edges
        not_forbidden = (s,e,k) not in forbidden_edges
        # for a pair of twin half-edge, it's possible for one to be in forbidden list and the other not
        # so if the occupancy suggests that they should be removed, one of them will be removed
        # this is problematic for arrangement._decompose, I can't let this happen! No sir!
        (ts,te,tk) = arrangement.graph[s][e][k]['obj'].twinIdx
        not_forbidden = not_forbidden and  ((ts,te,tk) not in forbidden_edges)
        
        # rule 2: low_edge_occupancy - below "low_occ_percent"
        neighbors_dis_val = arrangement.graph[s][e][k]['obj'].attributes['distances']
        sum_ = neighbors_dis_val.sum() / 255.
        siz_ = np.max([1,neighbors_dis_val.shape[0]])
        ok_to_prun_sum = float(sum_)/siz_ > distance_threshold
        # var_ = np.var(neighbors_dis_val)
        # ok_to_prun_var = var_ < 200
        
        if not_forbidden and ok_to_prun_sum:
            edges_to_purge.append( (s,e,k) )

    # prunning
    arrangement.remove_edges(edges_to_purge, loose_degree=2)
    arrangement.remove_nodes(nodes_idx=[], loose_degree=2)

    return arrangement

################################################################################
def loader (png_name, n_categories=2):
    ''' Load files '''
    
    yaml_name = png_name[:-3] + 'yaml'
    skiz_name = png_name[:-4] + '_skiz.png'
    ply_name = png_name[:-3] + 'ply'
    label_name = png_name[:-4]+'_labels_km{:s}.npy'.format(str(n_categories))    
    
    dis_name = png_name[:-4] + '_dis.png'
    dis_name = png_name[:-4] + '_dis2.png'

    ### loading image and converting to binary 
    image = np.flipud( cv2.imread( png_name, cv2.IMREAD_GRAYSCALE) )
    thr1,thr2 = [200, 255]
    ret, image = cv2.threshold(image.astype(np.uint8) , thr1,thr2 , cv2.THRESH_BINARY)

    ### loading label_image
    label_image = np.load(label_name)

    ### loading distance image
    dis_image = np.flipud( cv2.imread(dis_name , cv2.IMREAD_GRAYSCALE) )

    ### laoding skiz image
    skiz = np.flipud( cv2.imread( skiz_name, cv2.IMREAD_GRAYSCALE) )    

    ### loading traits from yamls
    trait_data = arr.utls.load_data_from_yaml( yaml_name )   
    traits = trait_data['traits']
    boundary = trait_data['boundary']
    boundary[0] -= 20
    boundary[1] -= 20
    boundary[2] += 20
    boundary[3] += 20

    ### trimming traits
    traits = arr.utls.unbound_traits(traits)
    traits = arr.utls.bound_traits(traits, boundary)

    return image, label_image, dis_image, skiz, traits
