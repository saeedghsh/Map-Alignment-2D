from __future__ import print_function

import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

new_paths = [
    u'../arrangement/',
    u'../place_categorization_2D',
    # u'/home/saesha/Dropbox/myGits/Python-CPD/'
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import time
import itertools 

import cv2
import numpy as np
import sympy as sym
import networkx as nx
import scipy
import scipy.ndimage
import sklearn.cluster
import collections
import numpy.linalg
import skimage.transform

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms

import arrangement.arrangement as arr
reload(arr)
import arrangement.utils as utls
reload(utls)
import arrangement.plotting as aplt
reload(aplt)
import place_categorization as plcat
reload(plcat)


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
    'HIH_tango':      dir_tango+'HIH_01_full/20170131135829.png',

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
    
    'F5_01_tango':    dir_tango+'F5_1/20170131132256.png',
    'F5_02_tango':    dir_tango+'F5_2/20170131125250.png',
    'F5_03_tango':    dir_tango+'F5_3/20170205114543.png',
    'F5_04_tango':    dir_tango+'F5_4/20170205115252.png',
    'F5_05_tango':    dir_tango+'F5_5/20170205115820.png'
}

################################################################################
def create_mpath ( points ):
    '''
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
    arrange0 = arrangements[arrangements.keys()[0]]
    arrange1 = arrangements[arrangements.keys()[1]]

    # find the alignment pool
    if face_similarity == 'vote':
        # find label association between maps - and remove label -1
        label_associations = label_association(arrangements, connectivity_maps)
        del label_associations[-1]

        tforms = np.array([])
        for lbl0 in label_associations.keys():
            lbl1 = label_associations[lbl0]

            # the first condition for "if" inside list comprehension rejects faces with undesired labels
            # the second condition rejects faces with area==0,
            # because if the area was zero, label_count was set all to zero
            faces0 = [ f_idx
                       for f_idx, face in enumerate(arrange0.decomposition.faces)
                       if (face.attributes['label_vote'] == lbl0) and (np.sum(face.attributes['label_count'].values())>0) ]
            faces1 = [ f_idx
                       for f_idx, face in enumerate(arrange1.decomposition.faces)
                       if (face.attributes['label_vote'] == lbl1) and (np.sum(face.attributes['label_count'].values())>0) ]

            for f0Idx, f1Idx in itertools.product(faces0, faces1):
                tfs_d = utls.align_faces( arrange0, arrange1,
                                          f0Idx, f1Idx,
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
        faces0 = [ f_idx
                   for f_idx, face in enumerate(arrange0.decomposition.faces)
                   if np.sum(face.attributes['label_count'].values()) > 0 ]
        faces1 = [ f_idx
                   for f_idx, face in enumerate(arrange1.decomposition.faces)
                   if np.sum(face.attributes['label_count'].values()) > 0 ]
        
        tforms = np.array([])
        for f0Idx, f1Idx in itertools.product(faces0, faces1):
            face0 = arrange0.decomposition.faces[f0Idx]
            face1 = arrange1.decomposition.faces[f1Idx]
            if are_same_category(face0, face1, label_associations, thr=.4):
                tfs_d = utls.align_faces( arrange0, arrange1,
                                          f0Idx, f1Idx,
                                          tform_type=tform_type,
                                          enforce_match=enforce_match)
                tforms = np.concatenate(( tforms,
                                          np.array([ tfs_d[k]
                                                     for k in tfs_d.keys() ]) ))

    elif face_similarity is None:
        ### find alignments among all faces, regardless of their labels

        # here again the condition for "if" inside list comprehension rejects faces with area==0,
        # because if the area was zero, label_count was set all to zero
        faces0 = [ f_idx
                   for f_idx, face in enumerate(arrange0.decomposition.faces)
                   if np.sum(face.attributes['label_count'].values()) > 0 ]
        faces1 = [ f_idx
                   for f_idx, face in enumerate(arrange1.decomposition.faces)
                   if np.sum(face.attributes['label_count'].values()) > 0  ]

        tforms = np.array([])
        for f0Idx, f1Idx in itertools.product(faces0, faces1):
            tfs_d = utls.align_faces(arrange0, arrange1,
                                     f0Idx, f1Idx,
                                     tform_type=tform_type,
                                     enforce_match=enforce_match)

            tforms = np.concatenate(( tforms,
                                      np.array([tfs_d[k] for k in tfs_d.keys()]) ))

    return tforms

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
            fs = [ (connectivity_maps[key].node[n_idx]['features'][0], # node degree
                    connectivity_maps[key].node[n_idx]['features'][3]) # node load centrality
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


    # finding associations between labels by minimum distance between their corresponding sets of feature
    # assuming label -1 is universal, we'll set it as default
    associations = {-1:-1}
    for lbl1 in labels:
        S1 = np.cov(  features['{:s}_{:d}'.format(keys[0], lbl1)], rowvar=False )
        U1 = np.mean( features['{:s}_{:d}'.format(keys[0], lbl1)], axis=0 )
        dist = [ bhattacharyya_distance (S1, U1,
                                         S2 = np.cov(  features['{:s}_{:d}'.format(keys[1], lbl2)], rowvar=False ),
                                         U2 = np.mean( features['{:s}_{:d}'.format(keys[1], lbl2)], axis=0 ) )
                 for lbl2 in labels ]

        idx = dist.index(min(dist))
        associations[lbl1] = labels[idx]

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
    if S1.shape !=(2,2): S1 = np.eye(2)
    if S2.shape !=(2,2): S2 = np.eye(2)

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
def assign_label_to_face(arrangement, label_image):
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
        in_face = face.path.contains_points(all_pixels)
        pixels = all_pixels[in_face, :]

        if pixels.shape[0]==0:
            label = -1
            labels = { lbl: 0.
                       for lbl in np.unique(label_image) }

        else:
            # mode=='vote'
            not_nan = np.nonzero( np.isnan(label_image[pixels[:,1],pixels[:,0]])==False )[0]
            label = np.median(label_image[pixels[:,1],pixels[:,0]][not_nan] )
            label = -1 if np.isnan(label) else label
        
            # mode=='count'
            total = float(pixels.shape[0])
            labels = { lbl: np.nonzero(label_image[pixels[:,1],pixels[:,0]]==lbl)[0].shape[0] /total
                       for lbl in np.unique(label_image)}
            # assert np.abs( np.sum([ labels[lbl] for lbl in labels.keys() ]) -1) < np.spacing(10**5)

        arrangement.decomposition.faces[idx].attributes['label_vote'] = label
        arrangement.decomposition.faces[idx].attributes['label_count'] = labels

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
    ''' '''
    dis = face_category_distance( face1,face2, label_associations )    
    return True if dis<thr else False


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
    faces = arrangement.decomposition.faces

    ########## node construction (one node per each face)
    nodes = [ [f_idx, {}] for f_idx,face in enumerate(faces) ]
    connectivity_map.add_nodes_from( nodes )

    # assuming convex faces, node coordinate = COG (face.nodes)
    if set_coordinates:
        for f_idx,face in enumerate(faces):
            nodes = [arrangement.graph.node[fn_idx]
                     for fn_idx in face.get_all_nodes_Idx()]
            x = np.mean([ node['obj'].point.x for node in nodes ])
            y = np.mean([ node['obj'].point.y for node in nodes ])
            # setting the c
            connectivity_map.node[f_idx]['coordinate'] = [x,y]
            arrangement.decomposition.faces[f_idx].attributes['centre'] = [x,y]

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
        
    ###### plottig - for the debuging and fine-tuning    
    internal_plotting = False
    if internal_plotting:

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 3)
        
        # image
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title('original')
        ax1.imshow(original, cmap = 'gray', interpolation='nearest', origin='lower')
        
        # image_binary
        ax2 = plt.subplot(gs[1, 0])
        ax2.set_title('image')
        ax2.imshow(image, cmap = 'gray', interpolation='nearest', origin='lower')
        
        # dis
        ax3 = plt.subplot(gs[0, 1])
        ax3.set_title('dis')
        ax3.imshow(dis, cmap = 'gray', interpolation='nearest', origin='lower')
        
        # grd_binary
        ax4 = plt.subplot(gs[1, 1])
        ax4.set_title('abs(grd) [binary_inv]')
        ax4.imshow(grd_abs, cmap = 'gray', interpolation='nearest', origin='lower')
        # ax4.imshow(grd_binary_inv, cmap = 'gray', interpolation='nearest', origin='lower')
        
        # voronoi
        ax5 = plt.subplot(gs[:, 2])
        ax5.set_title('skiz')
        ax5.imshow(skiz, cmap = 'gray', interpolation='nearest', origin='lower')

    plt.show()


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
def get_edges_to_prune (arrangement,
                        low_occ_percent, high_occ_percent,
                        consider_categories=True):
    '''
    This method returns a list of edges to be removed from the arrangement.
    Edge pruning rules:
    - rule 1: (self and twin) not in forbidden_edges
    - rule 2: low_edge_occupancy - below "low_occ_percent"
    - rule 3: not_high_node_occupancy - not more than "high_occ_percent"
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
    edges_to_prune: list of tuples [(s,e,k), ... ]

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

    edges_to_prune = []
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
            edges_to_prune.append( (s,e,k) )
    
    for (s,e,k) in edges_to_prune:
        if arrangement.graph[s][e][k]['obj'].twinIdx not in edges_to_prune:
            raise (NameError('there is a half-edge whos twin is not in the "edges_to_prune"'))

    return edges_to_prune


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
def arrangement_pruning(arrangement, image,
                        image_occupancy_thr = 200,
                        edge_neighborhood = 10,
                        node_neighborhood = 10,
                        low_occ_percent  = .025, # .005 # .01# .02 - below "low_occ_percent"
                        high_occ_percent = .1, # .050 # .03# .04 - not more than "high_occ_percent"
                        consider_categories = True
                       ): 
    '''
    category_forbidden
    It is passed to "get_edges_to_prune"
    Is category_forbidden is True, "forbidden_edges" in the "get_edges_to_prune"
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
    # edge pruning:
    edges_to_prune = get_edges_to_prune (arrangement,
                                         low_occ_percent, high_occ_percent,
                                         consider_categories)
    arrangement.remove_edges(edges_to_prune, loose_degree=2)
    
    # node pruning:
    nodes_to_prune = get_nodes_to_prune (arrangement,
                                         low_occ_percent, high_occ_percent)
    arrangement.remove_nodes(nodes_to_prune, loose_degree=2)
    
    return arrangement




################################################################################
def loader (png_name, n_categories=2):
    ''' Load files '''
    
    yaml_name = png_name[:-3] + 'yaml'
    skiz_name = png_name[:-4] + '_skiz.png'
    ply_name = png_name[:-3] + 'ply'
    label_name = png_name[:-4]+'_labels_km{:s}.npy'.format(str(n_categories))    


    ### loading image and converting to binary 
    image = np.flipud( cv2.imread( png_name, cv2.IMREAD_GRAYSCALE) )
    thr1,thr2 = [200, 255]
    ret, image = cv2.threshold(image.astype(np.uint8) , thr1,thr2 , cv2.THRESH_BINARY)

    ### loading label_image
    label_image = np.load(label_name)

    ### laoding skiz image
    skiz = np.flipud( cv2.imread( skiz_name, cv2.IMREAD_GRAYSCALE) )    

    ### loading traits from yamls
    trait_data = utls.load_data_from_yaml( yaml_name )   
    traits = trait_data['traits']
    boundary = trait_data['boundary']
    boundary[0] -= 20
    boundary[1] -= 20
    boundary[2] += 20
    boundary[3] += 20

    ### trimming traits
    traits = utls.unbound_traits(traits)
    traits = utls.bound_traits(traits, boundary)

    return image, label_image, skiz, traits




################################################################################
################################################# plotting stuff - not important
################################################################################

def plot_connectivity_map(axes, connectivity_map):
    X,Y = zip( *[ connectivity_map.node[key]['coordinate']
                  for key in connectivity_map.node.keys() ] )
    axes.plot(X,Y, 'go', alpha=.7)
    for (s,e,k) in connectivity_map.edges(keys=True):
        X,Y = zip( *[ connectivity_map.node[key]['coordinate']
                      for key in (s,e) ] )
        axes.plot(X,Y, 'g-', alpha=.7)
    
    return axes

########################################
def plot_image(axes, image, alpha=1., cmap='gray'):
    axes.imshow(image, cmap=cmap, alpha=alpha, interpolation='nearest', origin='lower')
    return axes

########################################
def plot_arrangement(axes, arrange, printLabels=False ):
    aplt.plot_edges (axes, arrange, alp=.1, col='b', printLabels=printLabels)
    aplt.plot_nodes (axes, arrange, alp=.5, col='r', printLabels=printLabels)
    return axes

################################################################################
def plot_place_categories (axes, arrangement, alpha=.5):

    clrs = ['k', 'm', 'y', 'c', 'b', 'r', 'g']
    
    for face in arrangement.decomposition.faces:
        # face.attributes['label'] = [-1,0,...,5]
        clr = clrs [ int(face.attributes['label_vote']+1) ]    
        patch = mpatches.PathPatch(face.get_punched_path(),
                                   facecolor=clr, edgecolor=None,
                                   alpha=alpha)        
        axes.add_patch(patch)


########################################
def plot_node_edge_occupancy_statistics(arrange, bins=30):
    ''' '''
    edge_occ = np.array([ arrange.graph[s][e][k]['obj'].attributes['occupancy']
                          for (s,e,k) in arrange.graph.edges(keys=True) ]).astype(float)

    node_occ = np.array([ arrange.graph.node[key]['obj'].attributes['occupancy']
                          for key in arrange.graph.nodes() ]).astype(float)


    fig, axes = plt.subplots(2,2, figsize=(20,12))
    axes[0,0].set_title('edge occupancy')
    axes[0,0].plot(edge_occ[:,0], label='#occupied')
    axes[0,0].plot(edge_occ[:,1], label='#neighbors')
    axes[0,0].plot(edge_occ[:,0] / edge_occ[:,1], label='#o/#n')
    axes[0,0].legend(loc='upper right')    

    # axes[1,0].set_title('edge occupancy')
    axes[1,0].hist(edge_occ[:,0] / edge_occ[:,1], bins=bins)

    axes[0,1].set_title('node occupancy')
    axes[0,1].plot(node_occ[:,0], label='#occupied')
    axes[0,1].plot(node_occ[:,1], label='#neighbors')
    axes[0,1].plot(node_occ[:,0] / node_occ[:,1], label='#o/#n')
    axes[0,1].legend(loc='upper right')

    # axes[1,1].set_title('node occupancy')
    axes[1,1].hist(node_occ[:,0] / node_occ[:,1], bins=bins)
    
    plt.tight_layout()
    plt.show()


########################################
def plot_point_sets (src, dst=None):
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    fig.axes[0].plot(src[:,0] ,  src[:,1], 'b.')
    if dst is not None:
        fig.axes[0].plot(dst[:,0] ,  dst[:,1], 'r.')
    fig.axes[0].axis('equal')
    # fig.show() # using fig.show() won't block the code!
    plt.show()

################################################################################
def visualize(X, Y, ax):
    '''
    This method is for the animation of CPD from:
    https://github.com/siavashk/Python-CPD
    '''

    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue')
    ax.axis('equal')
    plt.draw()
    plt.pause(0.01**5)


################################################################################
def plot_trnsfromed_images(src, dst, tformM=None ):
    '''
    tformM - 2darray (3x3)
    default (None) will result in an identity matrix
    '''

    aff2d = matplotlib.transforms.Affine2D( tformM )
    # aff2d._mtx == tformM
    
    fig, axes = plt.subplots(1,1, figsize=(20,12))

    # drawing images and transforming src image
    im_dst = axes.imshow(dst, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    im_src = axes.imshow(src, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    im_src.set_transform( aff2d + axes.transData )

    # finding the extent of of dst and transformed src
    xmin_d,xmax_d, ymin_d,ymax_d = im_dst.get_extent()
    x1, x2, y1, y2 = im_src.get_extent()
    pts = [[x1,y1], [x2,y1], [x2,y2], [x1,y1]]
    pts_tfrom = aff2d.transform(pts)    

    xmin_s, xmax_s = np.min(pts_tfrom[:,0]), np.max(pts_tfrom[:,0]) 
    ymin_s, ymax_s = np.min(pts_tfrom[:,1]), np.max(pts_tfrom[:,1])

    # setting the limits of axis to the extents of images
    axes.set_xlim( min(xmin_s,xmin_d), max(xmax_s,xmax_d) )
    axes.set_ylim( min(ymin_s,ymin_d), max(ymax_s,ymax_d) )

    # # turn of tickes
    # axes.set_xticks([])
    # axes.set_yticks([])

    plt.tight_layout()
    plt.show()


################################################################################
def histogram_of_face_category_distances(arrangement):
    '''
    this method plots the histogram of face category distance (face.attributes['label_count'])
    
    Blue histogram:
    distances between those faces that are assigned with same category in voting (face.attributes['label_vote'])

    Red hsitogram:
    distances between those faces that are assigned with differnt categories in voting (face.attributes['label_vote'])

    '''
    same_lbl_dis = []
    diff_lbl_dis = []
    for (f1,f2) in itertools.combinations( arrangement.decomposition.faces, 2):
    
        dis = face_category_distance(f1,f2)

        if f1.attributes['label_vote'] == f2.attributes['label_vote']:
            same_lbl_dis += [dis]
        else:
            diff_lbl_dis += [dis]
    
    fig, axes = plt.subplots(1,1, figsize=(20,12))
    h_same = axes.hist(same_lbl_dis, facecolor='b', bins=30, alpha=0.7, label='same category')
    h_diff = axes.hist(diff_lbl_dis, facecolor='r', bins=30, alpha=0.7, label='diff category')

    axes.legend(loc=1, ncol=1)
    axes.set_title('histogram of face category distance')
    plt.tight_layout()
    plt.show()


################################################################################
def histogram_of_alignment_parameters(parameters):
    '''
    '''    
    fig, axes = plt.subplots(1,1, figsize=(20,12))
    axes.hist(parameters[:,0], facecolor='b', bins=1000, alpha=0.7, label='tx')
    axes.hist(parameters[:,1], facecolor='r', bins=1000, alpha=0.7, label='ty')
    axes.hist(parameters[:,2], facecolor='g', bins=1000, alpha=0.7, label='rotate')
    axes.hist(parameters[:,3], facecolor='m', bins=1000, alpha=0.7, label='scale')

    axes.legend(loc=1, ncol=1)
    axes.set_title('histogram of alignment parameters')
    plt.tight_layout()
    plt.show()
