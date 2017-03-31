from __future__ import print_function

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

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms

import arrangement.arrangement as arr
# reload(arr)
import arrangement.utils as utls
# reload(utls)
import arrangement.plotting as aplt
# reload(aplt)
# import place_categorization as plcat
# reload(plcat)




################################################################################
################################################################ out-dated stuff
################################################################################


################################################################################
def find_face2face_association(arrangement_src, arrangement_dst,
                               distance=['area','centre'][0]):
    '''
    problems:
    this does not result in a one to one assignment
    '''

    # set center coordinate for faces
    set_face_centre_attribute(arrangement_src)
    set_face_centre_attribute(arrangement_dst)
    face_cen_src = np.array([face.attributes['centre']
                             for face in arrangement_src.decomposition.faces])
    face_cen_dst = np.array([face.attributes['centre']
                             for face in arrangement_dst.decomposition.faces])    

    if distance=='area':
        face_area_src = np.array([face.get_area()
                                  for face in arrangement_src.decomposition.faces])
        face_area_dst = np.array([face.get_area()
                                  for face in arrangement_dst.decomposition.faces])
        # cdist expects 2d arrays as input, so I just convert the 1d area value
        # to 2d vectors, all with one (or any aribitrary number)
        face_area_src_2d = np.stack((face_area_src, np.ones((face_area_src.shape))),axis=1)
        face_area_dst_2d = np.stack((face_area_dst, np.ones((face_area_dst.shape))),axis=1)
        f2f_distance = scipy.spatial.distance.cdist(face_area_src_2d,
                                                    face_area_dst_2d,
                                                    'euclidean')
    elif distance=='centre':
        f2f_distance = scipy.spatial.distance.cdist(face_cen_src,
                                                    face_cen_dst,
                                                    'euclidean')
        
    f2f_association = {}
    for src_idx in range(f2f_distance.shape[0]):
        # if the centre of faces in dst are not inside the current face of src
        # their distance are set to max, so that they become illegible
        # TODO: should I also check for f_dst.path.contains_point(face_cen_src)

        f_src = arrangement_src.decomposition.faces[src_idx]
        contained_in_src = f_src.path.contains_points(face_cen_dst)
        contained_in_dst = [f_dst.path.contains_point(face_cen_src[src_idx])
                            for f_dst in arrangement_dst.decomposition.faces]
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
def arrangement_match_score(arrangement1, arrangement2):
    '''
    '''

    # find the area of each arrangement
    superface1 = arrangement1._get_independent_superfaces()[0]
    superface2 = arrangement2._get_independent_superfaces()[0]

    arrangement1_area = superface1.get_area()
    arrangement2_area = superface2.get_area()

    # find face to face association
    f2f_association = find_face2face_association(arrangement1,
                                                 arrangement2,
                                                 distance='area')

    # find face to face match score (of associated faces)
    f2f_match_score = {(f1_idx,f2f_association[f1_idx]): None
                       for f1_idx in f2f_association.keys()}
    for (f1_idx,f2_idx) in f2f_match_score.keys():
        score = face_match_score(arrangement1.decomposition.faces[f1_idx],
                                 arrangement2.decomposition.faces[f2_idx])
        f2f_match_score[(f1_idx,f2_idx)] = score

    # find the weights of pairs of associated faces to arrangement match score
    face_pair_weight = {}
    for (f1_idx,f2_idx) in f2f_match_score.keys():
        f1_area = arrangement1.decomposition.faces[f1_idx].get_area()
        f2_area = arrangement2.decomposition.faces[f2_idx].get_area()

        f1_w = float(f1_area) / float(arrangement1_area)
        f2_w = float(f2_area) / float(arrangement2_area)

        face_pair_weight[(f1_idx,f2_idx)] = np.min([f1_w, f2_w])

    # computing arrangement match score
    arr_score = np.sum([face_pair_weight[(f1_idx,f2_idx)]*f2f_match_score[(f1_idx,f2_idx)]
                        for (f1_idx,f2_idx) in f2f_match_score.keys()])

    return arr_score


################################################################################
def pointset_match_score (src,dst,sigma, tform=None):
    '''
    This method constructs a set of gaussian distributions, each centered at the
    location of points in destination, with a diagonal covariance matix of sigma.
    It evaluates the values of all points in src in all gaussian models.
    The match score is the sum of all evaluation devided by the number of points
    in the src point set.
    '''
    if tform is None: tform = skimage.transform.AffineTransform()

    S = np.array([[sigma,0] , [0,sigma]])
    src_warp = tform._apply_mat(src, tform.params)

    match_score = np.array([ normal_dist(src_warp, M, S) for M in dst ])
    match_score = match_score.sum()
    match_score /= src.shape[0]

    return match_score

################################################################################
def objectivefun_pointset (X , *arg):
    '''
    X: the set of variables to be optimized
    
    src: source point set (model - template)
    dst: destination point set (static scene - target)

    sigma: variance of the normal distributions located at each destination points 
    '''
    tx, ty, s, t = X 
    tform = skimage.transform.AffineTransform(scale=(s,s),
                                              rotation=t,
                                              translation=(tx,ty))

    src, dst, sigma = arg[0], arg[1], arg[2]
    match_score = pointset_match_score (src,dst,sigma, tform)
    error = 1-match_score

    print (error)
    return error

################################################################################
def create_svgpath_from_mbb(image):
    '''
    Assuming the image starts at origin, and is aligned with main axes, this 
    method returns a svg-path around the image
    
    example
    -------
    path = create_svgpath_from_mbb(img)
    path.area()
    '''
    import svgpathtools
    minX, maxX = 0, image.shape[1]
    minY, maxY = 0, image.shape[0]
    path = svgpathtools.Path( svgpathtools.Line(minX+minY*1j, maxX+minY*1j),
                              svgpathtools.Line(maxX+minY*1j, maxX+maxY*1j),
                              svgpathtools.Line(maxX+maxY*1j, minX+maxY*1j),
                              svgpathtools.Line(minX+maxY*1j, minX+minY*1j) )
    assert path.isclosed()
    return path




################################################################################
def normal_dist_single_input (X, M, S):
    '''
    Inputs
    ------
    X: input point (1xd), where d is the dimension
    M: center of the normal distribution (1xd)
    S: covariance matrix of the normal distribution (dxd)

    Output
    ------
    res (scalar, float)    

    example
    -------
    >>> X = np.array([11,12])
    >>> M = np.array([10,10])
    >>> S = np.array([ [.5,0] , [0,.5] ])
    >>> print normal_dist_single_input(X,M,S)
    0.00214475514
    '''

    if np.abs(np.linalg.det(S)) < np.spacing(10):
        return None

    dis = X-M
    # dis is 1xd vector, transpose of normal math convention
    # that's why the transposes in the nomin are exchanged 
    nomin = np.exp( -.5 * np.dot( np.dot( dis, np.linalg.inv(S)), dis) )
    denom = np.sqrt( np.linalg.det( 2* np.pi* S) )
    res = nomin/denom
    return res


################################################################################
def normal_dist (X, M, S):
    '''
    This method evaluates the value of a normal distribution at every point
    in the input array, and returns as many values in output

    Inputs
    ------
    X: input points (nxd), where n is the number of points and d is the dimension
    M: center of the normal distribution (1xd)
    S: covariance matrix of the normal distribution (dxd)

    Output
    ------
    res (ndarray: 1xn)
    normal distribution's value at each input point

    example
    -------
    >>> X = np.array([ [11,12], [13,14],  [15,16],  [17,18] ])
    >>> M = np.array( [10,10] )
    >>> S = np.array([ [.5,0] , [0,.5] ])
    >>> print normal_dist(X,M,S)
    [  2.14475514e-03   4.42066983e-12   1.02538446e-27   2.67653959e-50]

    Note
    ----
    It won't work if X is (1xd)

    '''
    if np.abs(np.linalg.det(S)) < np.spacing(10):
        return None

    dis = X-M
    nomin = np.exp( -.5* ( np.dot( dis, np.linalg.inv(S)) * dis ).sum(axis=1) )
    denom = np.sqrt( np.linalg.det(2* np.pi* S) )
    res = nomin/denom
    return res

################################################################################
def aff_mat(scale, rotation, translation):
    sx,sy = scale
    t = rotation
    tx,ty = translation
    M = np.array([ [sx*np.cos(t), -sy*np.sin(t), tx*sx*np.cos(t) - ty*sy*sin(t)],
                   [sx*np.sin(t),  sy*np.cos(t), tx*sx*np.sin(t) + ty*sy*cos(t)],
                   [0,             0,            1                             ] ])
    return M

########################################
def rotation_mat_sym(rotation=None):
    t = sym.Symbol('t') if rotation is None else rotation
    R = np.array([ [sym.cos(t), -sym.sin(t), 0],
                   [sym.sin(t),  sym.cos(t), 0],
                   [0,           0,          1] ])
    return R
########################################
def scale_mat_sym(scale=None):
    sx,sy = (sym.Symbol('sx'),sym.Symbol('sy') ) if scale is None else scale
    S = np.array([ [sx, 0,  0],
                   [0,  sy, 0],
                   [0,  0,  1] ])
    return S
########################################
def translation_mat_sym(translation=None):
    tx,ty = (sym.Symbol('tx'), sym.Symbol('ty') ) if translation is None else translation
    T = np.array([ [1, 0, tx],
                   [0, 1, ty],
                   [0, 0, 1 ] ])
    return T


################################################################################
def get_pixels_in_cirlce(pixels, centre, radius):    
    '''
    given a set of pixels, returns those in inside the define circle
    '''    
    ####### find pixels_in_circle:
    # cdist expects post input to be 2d, hence the use of np.atleast_2d
    dists = scipy.spatial.distance.cdist(np.atleast_2d(centre), pixels, 'euclidean')
    # flagging all the points that are in the circle
    # due to cdist's result, the first element on nonzero is a sequence of zeros
    # the first element is an array of row indices, and since dists is 1xx -> rows=0
    pixel_inbound_idx = np.nonzero( dists<=radius )[1]
    return pixels[pixel_inbound_idx]
    

################################################################################
def categorize_faces (image, arrangement,
                      radius = None,
                      n_categories = 2,
                      mpp = 0.02, # meter per pixel
                      range_meter = 8, # meter
                      length_range = 400, #range_meter_ / mpp_
                      length_steps = 200, #int(length_range_)
                      theta_range = 2*np.pi,
                      theta_res = 1/1, # step/degree
                      occupancy_thr = 180,
                      gapThreshold = [1.0]
                  ):
    '''
    raycast from the center of each face, within a circle of "radius"
    '''
    # erode the ogm to make it suitable for raycasting
    kernel = np.ones((5,5),np.uint8)
    raycast_image = cv2.erode(image, kernel, iterations = 1)
    raycast_image = cv2.medianBlur(raycast_image, 5)

    ########## raycast template
    pose_ = np.array([0,0,0]) # x,y,theta
    rays_array_xy = plcat.construct_raycast_array(pose_,
                                                  length_range, length_steps, 
                                                  theta_range, theta_res)
    raxy = rays_array_xy


    ##########
    # storage of the features and their corresponding faces 
    face_idx = np.array([-1])
    features = np.zeros(( 1, 16+len(gapThreshold) ))

    ########## feature extraction per face
    for f_idx, face in enumerate(arrangement.decomposition.faces):
        pc = face.attributes['centre']
        centre = np.array([pc[0],pc[1]])

        # Note that after pruning, can't be sure if the center of face is inside!
        mbb = get_pixels_in_mpath(face.path, raycast_image.shape)
        if radius is None:
            inbounds = mbb
        else:
            inbounds = get_pixels_in_cirlce(mbb, centre, radius)
        
        ###### finding open-cells among pixels_in_circle_in_face
        open_cells_idx = np.nonzero( raycast_image[inbounds[:,1],inbounds[:,0]] >= occupancy_thr+15 )[0]
        open_cells = inbounds[open_cells_idx]


        # if the numebr of open_cells is too small, means that
        # the chosen open_space is more like a small pocket
        # so it won't result any proper raycast, svd might fail
        mimimum_openspace_size = 5
        if open_cells.shape[0] > mimimum_openspace_size:

            # per point in the open cells, a feature of len: 16+len(gapThreshold_)
            feat = np.zeros( ( len(open_cells), 16+len(gapThreshold) ) )
            # per pixels, add a new entry to face_idx array, corresponding to face index
            fc_ix = np.ones( len(open_cells) ) * int(f_idx)

            # raycasting and feature extraction from each pixel in open_cells
            for p_idx,(x,y) in enumerate(open_cells):

                pose_ = np.array([x,y,0]) # x,y,theta

                r,t = plcat.raycast_bitmap(raycast_image, pose_,
                                           occupancy_thr,
                                           length_range, length_steps, 
                                           theta_range, theta_res,
                                           rays_array_xy=raxy)

                feat[p_idx,:] = plcat.raycast_to_features(t,r,
                                                          mpp=mpp,
                                                          RLimit=range_meter,
                                                          gapThreshold=gapThreshold)

            features = np.concatenate( (features, feat) , axis=0)
            face_idx = np.concatenate( (face_idx, fc_ix) )

        
    ########## clustering
        
    ### feature modification
    X = features
    ## Normalizing 
    for i in range(X.shape[1]):
        X[:,i] /= X[:,i].mean()
    # rejectin NaNs
    X = np.where ( np.isnan(X), np.zeros(X.shape) , X) 
    
    ### clustering 
    kmean = sklearn.cluster.KMeans(n_clusters=n_categories,
                                   precompute_distances=False,
                                   n_init=20, max_iter=500)
    kmean.fit(X)
    labels = kmean.labels_
    
    
    ########## face labeling with voting
    for f_idx in np.arange(len(arrangement.decomposition.faces)):
        idx = np.nonzero(face_idx==f_idx)
        # if a face_center was rejected as pocket, its label is -1
        if len(idx[0]) > 0:
            l = np.median(labels[idx])
        else:
            l = -1
        arrangement.decomposition.faces[f_idx].attributes['label'] = l

    return arrangement

################################################################################
def get_nodes_to_prune (arrangement, low_occ_percent, high_occ_percent):
    '''
    This method returns a list of nodes to be removed from the arrangement.
    Node pruning rules:
    - rule 1: low node occupancy | below "low_occ_percent"
    - rule 2: no edge with high edge_occupancy | not more than "high_occ_percent"
    - rule 3: connected to no edge in forbidden edges
    for the explanation of forbidden_edges, see the note below
    
    Input
    -----
    arrangement

    Parameter
    ---------
    low_occ_percent: float between (0,1)
    high_occ_percent: float between (0,1)
    For their functionality see the description of the rules.
    Their value comes from the histogram of the occupancy ratio of the nodes and edges.
    execute "plot_node_edge_occupancy_statistics(arrangement) to see the histograms.
    I my experience, there are two peaks in the histograms, one close adjacent to zero
    and another one around .05.
    low_occ_percent is set slighlty after the first peak (~.02)
    high_occ_percent is set slighlty before the second peak (~.04)

    Output
    ------
    nodes_to_prune:
    list of node keys


    Note
    ----
    This method uses the occupancy values stored in the edges and nodes' attribute
    arrangement.graph[s][e][k]['obj'].attributes['occupancy']
    arrange.graph.node[s]['obj'].attributes['occupancy']
    Before calling this method, make sure they are set.

    Note
    ----
    fobidden_edge: edges that belong to the baoundary of the arrangement
    their removal will open up faces, hence undesired.
    must be checked in both edge-prunning and ALSO node prunning
    removing a node that will remove a forbidden edge is undesired
    the forbidden nodes are reflected in the forbidden edges, i.e a node that its
    removal would remove a forbidden edge.
    '''
    forbidden_edges  = arrangement.get_boundary_halfedges()
    nodes_to_prune = []
    for n_idx in arrangement.graph.node.keys():

        # rule 1
        o,n = arrangement.graph.node[n_idx]['obj'].attributes['occupancy']
        node_is_open = float(o)/n < low_occ_percent

        # rules 2 and 3
        edges_are_open = True
        edges_not_forbiden = True
        for (s,e,k) in arrangement.graph.out_edges([n_idx], keys=True):
            edges_not_forbiden = edges_not_forbiden and ((s,e,k) not in forbidden_edges)
            o, n = arrangement.graph[s][e][k]['obj'].attributes['occupancy']
            edges_are_open = edges_are_open and (float(o)/n < high_occ_percent)

        if node_is_open and edges_are_open and edges_not_forbiden:
            nodes_to_prune += [n_idx]

    return nodes_to_prune


################################################################################
def get_edges_to_purge (arrangement,
                        low_occ_percent, high_occ_percent,
                        consider_categories=True):
    '''
    This method returns a list of edges to be removed from the arrangement.
    Edge pruning rules:
    - rule 1: (self and twin) not in forbidden_edges
    - rule 2: low_edge_occupancy - self below "low_occ_percent"
    - rule 3: not_high_node_occupancy - no nodes with more than "high_occ_percent"
    for the explanation of forbidden_edges, see the note below    

    Input
    -----
    arrangement

    Parameter
    ---------
    low_occ_percent: float between (0,1)
    high_occ_percent: float between (0,1)
    For their functionality see the description of the rules.
    Their value comes from the histogram of the occupancy ratio of the nodes and edges.
    execute "plot_node_edge_occupancy_statistics(arrangement) to see the histograms.
    I my experience, there are two peaks in the histograms, one close adjacent to zero
    and another one around .05.
    low_occ_percent is set slighlty after the first peak (~.02)
    high_occ_percent is set slighlty before the second peak (~.04)

    consider_categories
    If consider_categories is True, "forbidden_edges" will include, in addition
    to boundary edges, those edges in between two faces with different place
    category labels.

    Output
    ------
    edges_to_purge: list of tuples [(s,e,k), ... ]

    Note
    ----
    This method uses the occupancy values stored in the edges and nodes' attribute
    arrangement.graph[s][e][k]['obj'].attributes['occupancy']
    arrange.graph.node[s]['obj'].attributes['occupancy']
    Before calling this method, make sure they are set.

    Note
    ----
    fobidden_edge: edges that belong to the baoundary of the arrangement
    their removal will open up faces, hence undesired.
    must be checked in both edge-prunning and ALSO node prunning
    removing a node that will remove a forbidden edge is undesired
    the forbidden nodes are reflected in the forbidden edges, i.e a node that its
    removal would remove a forbidden edge.
    '''

    forbidden_edges  = arrangement.get_boundary_halfedges()

    ###
    if consider_categories:
        low_occ_percent *= 10#4
        # high_occ_percent *= .5
        
        for (f1Idx, f2Idx) in itertools.combinations( range(len(arrangement.decomposition.faces)), 2):
            face1 = arrangement.decomposition.faces[f1Idx]
            face2 = arrangement.decomposition.faces[f2Idx]
            # note that we are comparing faces inside the same arrangment
            # so the label_associations is None (ie direct association)
            # with higher thr (thr=.8), I'm being generous on similarity 
            if not( are_same_category(face1,face2, label_associations=None, thr=.4) ):
            # if ( face1.attributes['label_vote'] != face2.attributes['label_vote']):
                forbidden_edges.extend( arrangement.decomposition.find_mutual_halfEdges(f1Idx, f2Idx) )

    # todo: raise an error if 'occupancy' is not in the attributes of the nodes and edges

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
        o, n = arrangement.graph[s][e][k]['obj'].attributes['occupancy']
        edge_is_open = float(o)/n < low_occ_percent
        
        # rule 3: not_high_node_occupancy - not more than "high_occ_percent"
        o,n = arrangement.graph.node[s]['obj'].attributes['occupancy']
        s_is_open = True #float(o)/n < high_occ_percent
        o,n = arrangement.graph.node[e]['obj'].attributes['occupancy']
        e_is_open = True #float(o)/n < high_occ_percent
                
        if not_forbidden and edge_is_open and (s_is_open or e_is_open):
            edges_to_purge.append( (s,e,k) )
    
    for (s,e,k) in edges_to_purge:
        if arrangement.graph[s][e][k]['obj'].twinIdx not in edges_to_purge:
            raise (NameError('there is a half-edge whos twin is not in the "edges_to_purge"'))

    return edges_to_purge


################################################################################
def set_node_occupancy(arrangement,
                       image, occupancy_thr=200,
                       neighborhood=10,
                       attribute_key='occupancy'):
    '''
    This method sets the occupancy every node in the arrangement wrt image:
    arrangement.graph.node[key]['obj'].attributes[attribute_key] = [occupied, neighborhood_area]

    Inputs
    ------
    arrangement
    The arrangement corresponding to the input image

    image: bitmap (gray scale)
    The image represensts the occupancy map and has high value for open space.

    Parameters
    ----------
    occupancy_thr: default:200
    Any pixel with value below "occupancy_thr" is considered occupied.

    neighborhood: default=10
    The radius of circle that defines the neighnorhood of a node

    attribute_key: default:'occupancy'
    The key to attribute dictionary of the nodes to store the "occupancy"
    This method is used for measuring occupancy of nodes against occupancy map and maybe other image
    Therefor it is important to store the result in the atrribute dictionary with proper key


    Note
    ----
    "neighborhood_area" is only dependant on the "neighborhood" parameter.
    Hence it is the same for all nodes
    '''

    # constructing a point set corresponding to occupied pixels 
    occupied_pixels = np.fliplr( np.transpose(np.nonzero(image<occupancy_thr)) )
    
    # approximately the number of pixels in the neighborhood disk
    neighborhood_area = np.pi * neighborhood**2
    
    for key in arrangement.graph.node.keys():
        # coordinates of the node
        p = np.array([ float(arrangement.graph.node[key]['obj'].point.x),
                       float(arrangement.graph.node[key]['obj'].point.y) ])

        # distance between the node and every point in tthe point set
        distance = np.sqrt( np.sum((occupied_pixels - p)**2, axis=1) )

        # counting occupied pixels in the neighborhood 
        occupied_neighbors = len( np.nonzero(distance<neighborhood)[0] )

        # if neighbors.shape[0] is zero, I will face division by zero
        # when checking for occupancy ratio. this is actually a problem
        # for short edges, not point's neighhborhood. Unless stupidly I
        # set neighborhood =0!
        o = occupied_neighbors
        n = np.max([1,neighborhood_area])

        arrangement.graph.node[key]['obj'].attributes[attribute_key] = [o, n]

################################################################################
def prune_arrangement( arrangement, image,
                       image_occupancy_thr = 200,
                       edge_neighborhood = 10,
                       node_neighborhood = 10,
                       low_occ_percent  = .025, # .005 # .01# .02 - below "low_occ_percent"
                       high_occ_percent = .1, # .050 # .03# .04 - not more than "high_occ_percent"
                       consider_categories = True
                   ): 
    '''
    category_forbidden
    It is passed to "get_edges_to_purge"
    Is category_forbidden is True, "forbidden_edges" in the "get_edges_to_purge"
    method will include, in addition to boundary edges, those edges in between
    two faces with different place category labels.  
    '''

    ### setting occupancy for nodes and edges
    set_node_occupancy(arrangement, image,
                       occupancy_thr=image_occupancy_thr,
                       neighborhood=node_neighborhood,
                       attribute_key='occupancy')

    set_edge_occupancy(arrangement, image,
                       occupancy_thr=image_occupancy_thr,
                       neighborhood=edge_neighborhood,
                       attribute_key='occupancy')

    ### prunning - source: occupancy         
    # edge purging:
    edges_to_purge = get_edges_to_purge (arrangement,
                                         low_occ_percent,
                                         high_occ_percent,
                                         consider_categories)
    arrangement.remove_edges(edges_to_purge, loose_degree=2)
    
    # node purging:
    nodes_to_prune = get_nodes_to_prune (arrangement,
                                         low_occ_percent, high_occ_percent)
    arrangement.remove_nodes(nodes_to_prune, loose_degree=2)
    
    return arrangement


# ################################################################################
# def prune_arrangement_with_face_growing (arrangement, label_image,
#                                          low_occ_percent, # for edge to prun
#                                          high_occ_percent, # for edge to prun
#                                          consider_categories, # for edge to prun
#                                          similar_thr):
#     '''
#     For every neighboruing faces, merge if:
#     1. faces are neighboring with same place categies
#     2. the mutual edges are in the list `edges_to_purge`
#     3. the emerging face must have the same shape as initial faces

#     Note
#     ----
#     Very very experimental:
#     1) arr.merge_faces_on_fly() is not complete, see the method itself
#     2) stochastic merging of faces and the condition on rejecting the emerging
#     face if the shape does not match the original, would lead to cases where 
#     the merging would not resolve the over decomposition:
#     |x|x|x|
#     |x|x|x|
#     |x|x|x|
#     to:
#     |0 0|1|
#     |3 x|1|
#     |3|2 2|
     
#     '''

#     # be almost generous with this, not too much, this is like a green light
#     # the faces won't merge unless they have same category and same shape 
#     edges_to_purge = get_edges_to_purge (arrangement,
#                                          low_occ_percent=low_occ_percent,
#                                          high_occ_percent=high_occ_percent,
#                                          consider_categories=consider_categories )

#     # set shape attribute for all faces
#     for face in arrangement.decomposition.faces:
#         face.set_shape_descriptor(arrangement, remove_redundant_lines=True)    

#     done_growing = False
#     while not done_growing:
#         # unless a new pair of faces are merged, we assume we are done merging
#         done_growing = True

#         faces = arrangement.decomposition.faces
#         for (f1Idx,f2Idx) in itertools.combinations( range(len(faces) ), 2):
    
#             f1, f2 = faces[f1Idx], faces[f2Idx]
#             mut_he = arrangement.decomposition.find_mutual_halfEdges(f1Idx,f2Idx)                    
#             # checking if faces are similar (category label)
#             similar_label = are_same_category(f1,f2, thr=similar_thr)
            
#             # checking if faces are similar (shape)
#             similar_shape = True if len(utls.match_face_shape(f1,f2))>0 else False
            
#             # cheking if it's ok to prun mutual halfedges
#             ok_to_prun = all([he in edges_to_purge for he in mut_he])
            
#             if similar_label and similar_shape and ok_to_prun:
            
#                 new_face = arr.merge_faces_on_fly(arrangement, f1Idx, f2Idx)
#                 # new_face could be none if faces are not neighbors, or
#                 # if there are disjoint chains of halfedges 
#                 if new_face is not None:
                    
#                     # checking if the new face has similar shape to originals
#                     new_face.set_shape_descriptor(arrangement, remove_redundant_lines=True)
#                     if len(utls.match_face_shape(new_face,f1))>0:
#                         done_growing = False
#                         # Note: change of faces tuple messes up the indices
#                         # and the correspondance with con_map nodes
                        
#                         # setting label attributes of the new_face
#                         assign_label_to_face(label_image, new_face)
                        
#                         # updating the list of faces
#                         fcs = list(faces)
#                         fcs.pop(max([f1Idx, f2Idx]))
#                         fcs.pop(min([f1Idx, f2Idx]))
#                         fcs.append(new_face)
#                         arrangement.decomposition.faces = tuple(fcs)
                        
#                         # removing edges from the graph
#                         arrangement.graph.remove_edges_from(mut_he)
                        
#                         # since the tuple of faces is changed, start over
#                         break

#     # remove redundant nodes
#     arrangement.remove_nodes(nodes_idx=[], loose_degree=2)
#     return arrangement


################################################################################
def construct_transformation_population(arrangements,
                                        connectivity_maps,
                                        face_similarity=['vote','count',None][0],
                                        tform_type=['similarity','affine'][0],
                                        enforce_match=False):
    '''
    Note
    ----
    connectivity_maps is used to find "label_association" between place categories
    
    if similarity None None connectivity_maps is not needed, since transformations
    among all faces, regardless of their labels, are returned.
    
    Note
    ----
    before calling this method the face.set_shape_descriptor() should be called
        
    '''
    arrange_src = arrangements[arrangements.keys()[0]]
    arrange_dst = arrangements[arrangements.keys()[1]]

    # find the alignment pool
    if face_similarity == 'vote':
        # find label association between maps - and remove label -1
        label_associations = label_association(arrangements, connectivity_maps)
        del label_associations[-1]

        tforms = np.array([])
        for lbl0 in label_associations.keys():

            if lbl0 == -1:
                pass # don't align faces with -1 category label
            else:
                lbl1 = label_associations[lbl0]
                
                # the first condition for "if" inside list comprehension rejects faces with undesired labels
                # the second condition rejects faces with area==0,
                # because if the area was zero, label_count was set all to zero
                faces_src = [ f_idx
                              for f_idx, face in enumerate(arrange_src.decomposition.faces)
                              if (face.attributes['label_vote'] == lbl0) and (np.sum(face.attributes['label_count'].values())>0) ]
                faces_dst = [ f_idx
                              for f_idx, face in enumerate(arrange_dst.decomposition.faces)
                              if (face.attributes['label_vote'] == lbl1) and (np.sum(face.attributes['label_count'].values())>0) ]

                for f_src_idx, f_dst_idx in itertools.product(faces_src, faces_dst):
                    tfs_d = utls.align_faces( arrange_src, arrange_dst,
                                              f_src_idx, f_dst_idx,
                                              tform_type=tform_type,
                                              enforce_match=enforce_match)
                    tforms = np.concatenate(( tforms,
                                              np.array([tfs_d[k] for k in tfs_d.keys()]) ))

    elif face_similarity == 'count':

        # find label association between maps - and remove label -1
        label_associations = label_association(arrangements, connectivity_maps)
        del label_associations[-1]

        # here again the condition for "if" inside list comprehension rejects faces with area==0,
        # because if the area was zero, label_count was set all to zero
        # the other condition rejects a face if it is mostly covered by uncategorized pixels (ie -1)
        faces_src = [ f_idx
                      for f_idx, face in enumerate(arrange_src.decomposition.faces)
                      if (face.attributes['label_count'][-1]<.1) and (np.sum(face.attributes['label_count'].values())>0) ]
        faces_dst = [ f_idx
                      for f_idx, face in enumerate(arrange_dst.decomposition.faces)
                      if (face.attributes['label_count'][-1]<.1) and (np.sum(face.attributes['label_count'].values())>0) ]
        
        tforms = np.array([])
        for f_src_idx, f_dst_idx in itertools.product(faces_src, faces_dst):
            face_src = arrange_src.decomposition.faces[f0Idx]
            face_dst = arrange_dst.decomposition.faces[f1Idx]
            if are_same_category(face_src, face_dst, label_associations, thr=.4):
                tfs_d = utls.align_faces( arrange_src, arrange_dst,
                                          f_src_idx, f_dst_idx,
                                          tform_type=tform_type,
                                          enforce_match=enforce_match)
                tforms = np.concatenate(( tforms,
                                          np.array([ tfs_d[k] for k in tfs_d.keys() ]) ))

    elif face_similarity is None:
        ### find alignments among all faces, regardless of their labels

        # here again the condition for "if" inside list comprehension rejects faces with area==0,
        # because if the area was zero, label_count was set all to zero
        faces_src = [ f_idx
                      for f_idx, face in enumerate(arrange_src.decomposition.faces)
                      if np.sum(face.attributes['label_count'].values()) > 0 ]
        faces_dst = [ f_idx
                      for f_idx, face in enumerate(arrange_dst.decomposition.faces)
                      if np.sum(face.attributes['label_count'].values()) > 0  ]

        tforms = np.array([])
        for f_src_idx, f_dst_idx in itertools.product(faces_src, faces_dst):
            tfs_d = utls.align_faces(arrange_src, arrange_dst,
                                     f_src_idx, f_dst_idx,
                                     tform_type=tform_type,
                                     enforce_match=enforce_match)

            tforms = np.concatenate(( tforms,
                                      np.array([tfs_d[k] for k in tfs_d.keys()]) ))

    return tforms
