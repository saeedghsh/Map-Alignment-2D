################################################################################
########################################################################### DUMP
################################################################################

def halfedge_neighborhood (arrangement, (s,e,k), neighborhood=5):
    '''
    neighborhood: window size in pixels (default:10)
    '''
    
    he = arrangement.graph[s][e][k]['obj']
    trait = arrangement.traits[he.traitIdx]
    str_node = arrangement.graph.node[s]
    end_node = arrangement.graph.node[e]
    (sTVal, eTVal) = he.get_tvals (arrangement.traits, arrangement.graph.node)

    # todo: why type(trait) is <type 'instance'>? not trts.xxModified?
    if isinstance(trait.obj, (sym.Segment, sym.Line, sym.Ray)):
        # take the length of the segment
        N = int(str_node['obj'].point.distance(end_node['obj'].point))
    elif isinstance(trait.obj, sym.Circle):
        # take the length of the arc
        N = int (2*trait.obj.radius*np.abs(eTVal-sTVal))
    
    # points in between will be covered as neighbor
    # but for big "neighborhood" it will become patchy!
    # N /= (neighborhood-1)
    N /= 3

    neighbors = []
    tval = np.linspace(float(sTVal), float(eTVal), num=N, endpoint=True )
    for t in tval:
        p = trait.DPE(t)
        if isinstance(p, sym.Point):
            neighbors.extend([ [p.y+row,p.x+col]
                               for row in range(-neighborhood+1,neighborhood) 
                               for col in range(-neighborhood+1,neighborhood)])

    # neighbors = np.array([ [p.y,px]
    #                        trait.DPE(t)
    #                        for t in np.linspace(sTVal, eTVal, num=N, endpoint=True) ])

    neighbors = np.array(neighbors)
    neighbors = np.vstack({tuple(row) for row in neighbors.astype(int)})        

    return neighbors





# '''
# [Wikipedia-Point_set_registration]
# In convergence, it is desired for the distance between the two point sets to reach a global minimum. This is difficult without exhausting all possible transformations, so a local minimum suffices.

# dist(T(M),S) = sum_{m\in T(M)} sum_{s\in S} (m-s)^2

# this function is sensitive to outlier data and consequently algorithms based on this function tend to be less robust against noisy data. A more robust formulation of the cost function uses some robust function g:

# dist_{robust}(T(M),S) = sum_{m\in T(M)} sum_{s\in S} g((m-s)^2)

# Such a formulation is known as an M-estimator. The robust function g is chosen such that the local configuration of the point set is insensitive to distant points, hence making it robust against outliers and noise


# RPM:

# approach 1:
# use ransac to estimate a similarity transform M := V1 \mapto V2
# V1 = [arrange1.graph.node[key]['obj'].point for key in arrange1.graph.node.keys()]
# V2 = [arrange2.graph.node[key]['obj'].point for key in arrange1.graph.node.keys()]

# approach 2:
# limit rotations by:
# theta := beta1 \mapto [alpha1, alpha1+pi alpha1, alpha1+pi]
# theta := beta2 \mapto [alpha1, alpha1+pi alpha1, alpha1+pi]
# [alpha1, alpha2] := dominant orientations of the map 1
# [beta1, beta2] := dominant orientations of map2
# then use ransac to find a translate-scale transform M:= V1_li \mapto V2_li
# V1_li = nodes belonging to the same line inf map 1
# V2_li = nodes belonging to the same line inf map 2

# Note:
# assert abs(beta2-beta1) == abs(alpha2-alpha1)
# if this not correct, the scaling between the maps won't be uniform!
# '''

# ######################################## approach 1
# if 1:
#     # src (Model set) is the tango map, ie arrange2
#     # dst (static Scene set) is the layout map, ie arrange1

#     src = [arrange2.graph.node[key]['obj'].point for key in arrange2.graph.node.keys()]
#     src = np.array([ [float(p.x),float(p.y)] for p in src ])
#     dst = [arrange1.graph.node[key]['obj'].point for key in arrange1.graph.node.keys()]
#     dst = np.array([ [float(p.x),float(p.y)] for p in dst ])

#     print ( src.shape, dst.shape )

#     # https://pypi.python.org/pypi/nudged/0.2.0
    
#     # ret = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=False)
#     # fullAffine – If true, the function finds an optimal affine transformation with no additional restrictions (6 degrees of freedom). Otherwise, the class of transformations to choose from is limited to combinations of translation, rotation, and uniform scaling (5 degrees of freedom).


#     ### shape matching based! very stupid
#     ### for a simple map of HIH, and assuming rectangle faces:
#     ### we have (171 * 220)*4 = 37620 *4 = 150480 potential combination to check!
#     # for f1Idx range(len(arrange1.decomposition.faces)):
#     #     for f2Idx range(len(arrange2.decomposition.faces)):
#     #         # f1 = arrang.decomposition.faces[f1Idx]
#     #         # f2 = arrang.decomposition.faces[f2Idx]
#     #         # print ( f1Idx, f2Idx, match_face_shape(f1,f2) )

#     ### trait matching! better than shape!


#     ### ransac based
#     tic = time.time()
#     max_itr = 10**5
#     for itr_idx in range(max_itr):
#         # step 1: sampling
#         # random sampels from src and dst (only 3 points)

#         src_smpl = src[ np.random.randint(low=0, high=src.shape[0], size=3)]
#         dst_smpl = dst[ np.random.randint(low=0, high=dst.shape[0], size=3)]

#         tform = skimage.transform.estimate_transform('similarity', src_smpl, dst_smpl)
#         # print (tform._matrix)

#         src_tform = tform(src)
#         res_err = sum(  )


#     print time.time() - tic
    
#     #     # step 2: transform estimate
#     #     # estimate transformation
    
#     #     # step 3: evaluated transformation
#     #     # transform the src, and find association between src_trs and dst
#     #     # metric 1: number of matched points  - metric 2: residual sum of squares of matched points


# ######################################## approach 2
# if 0:

#     ### layout map - dominant orientations
#     slopes1 = [ trait.obj.slope # note that the "trait.obj.slope" is a sympy object,
#                 for trait in traits1
#                 if isinstance(trait.obj, (sym.Line,sym.Segment,sym.Ray)) ]
#     angles1 = np.arctan(np.array(slopes1).astype(np.float))
#     # ori = scipy.stats.mode(angles1, axis=None)
#     orientations1 = list(set(angles1)) # assuming only two orientations
#     if len(orientations1) > 2:
#         print ( '\t Warning: there are more than 2 orientations in the map' )
#         for idx1 in range(len(orientations1)-1,-1,-1):
#             for idx2 in range(idx1):
#                 if np.abs(orientations1[idx1] - orientations1[idx2]) < np.spacing(10**10):
#                     orientations1.pop(idx1)
#                     break

#     ### tango map - dominant orientations
#     slopes2 = [ trait.obj.slope # note that the "trait.obj.slope" is a sympy object,
#                 for trait in traits2
#                 if isinstance(trait.obj, (sym.Line,sym.Segment,sym.Ray)) ]
#     angles2 = np.arctan(np.array(slopes2).astype(np.float))
#     # ori = scipy.stats.mode(angles2, axis=None)
#     orientations2 = list(set(angles2)) # assuming only two orientations
#     if len(orientations2) > 2:
#         print ( '\t Warning: there are more than 2 orientations in the map' )
#         for idx1 in range(len(orientations2)-1,-1,-1):
#             for idx2 in range(idx1):
#                 if np.abs(orientations2[idx1] - orientations2[idx2]) < np.spacing(10**10):
#                     orientations2.pop(idx1)
#                     break

#     ### testing "orthogonality!"
#     map1_is_orthogonal = np.abs(np.diff(orientations1)[0]) - np.pi/2 < np.spacing(10**10)
#     map2_is_orthogonal = np.abs(np.diff(orientations2)[0]) - np.pi/2 < np.spacing(10**10)
#     if not(map1_is_orthogonal) or not(map2_is_orthogonal):
#         print ('Warning: the assumption of orthogonal vector basis is not true...')    


#     ### findign rotation candidates
#     rotation_cand =  np.array([ orientations1[0],
#                                 orientations1[0]+np.pi,
#                                 orientations1[1],
#                                 orientations1[1]+np.pi] ) - orientations2[0]
#     print (rotation_cand)
