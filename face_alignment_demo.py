from __future__ import print_function

import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

import time
# import itertools 
import numpy as np
import sympy as sym
import networkx as nx
import matplotlib.pyplot as plt
import re

import itertools
import skimage.transform
import time

sys.path.append('../arrangement/')
import arrangement.arrangement as arr
reload(arr)
import arrangement.geometricTraits as trts
reload(trts)
import arrangement.plotting as aplt
reload(aplt)
import arrangement.utils as utls
reload(utls)

######################################## initiate traits
traits = []
N = 3
traits += [ trts.LineModified(args = (sym.Point(x,0),sym.Point(x,1))) # vertical lines
            for x in range(N) ]
traits += [ trts.LineModified(args = (sym.Point(0,y),sym.Point(1,y))) # horizontal lines
            for y in range(N) ]

traits += [ trts.SegmentModified(args = (sym.Point(0.5,1.5), sym.Point(1.5,1.5))) ]
traits += [ trts.CircleModified(args = (sym.Point(.5,.5),.3)) ]
traits += [ trts.CircleModified(args = (sym.Point(2,2),.2)) ]

######################################## construct arrangement
config = {'multi_processing':4, 'end_point':False, 'timing':True}
arrange = arr.Arrangement(traits, config)


######################################## plottings
print ('\t animation...')
aplt.animate_face_patches(arrange, timeInterval = .5* 1000)

print ('\t static plotting... ')
aplt.plot_decomposition(arrange,
                         # invert_axis = ('y'),
                         interactive_onClick=False,
                         interactive_onMove=False,
                         plotNodes=True, printNodeLabels=True,
                         plotEdges=True, printEdgeLabels=True)


######################################## testing area

#### setting face attribute with shape description
for idx,face in enumerate(arrange.decomposition.faces):
    arrange.decomposition.faces[idx].set_shape_descriptor(arrange)
    # print (face.attributes)

#### finding matches between two faces
print (3 * '\t ----------')
for (f1Idx,f2Idx) in itertools.combinations( range(len(arrange.decomposition.faces) ), 2):
    f1 = arrange.decomposition.faces[f1Idx]
    f2 = arrange.decomposition.faces[f2Idx]
    print ( f1Idx, f2Idx, utls.match_face_shape(f1,f2) )


#### finding alignment between two faces
print (3 * '\t ----------')
f1Idx,f2Idx = 0,1#,2 #6,8
arrange1, arrange2 = arrange, arrange
alignment_tforms = utls.align_faces(arrange1, arrange2, f1Idx, f2Idx)
for key in alignments.keys():
    print (key, alignment_tforms[key])

# alignment_tforms[key].rotation
# alignment_tforms[key].scale
# alignment_tforms[key].translation

# alignment_tforms[key]._inv_matrix
# alignment_tforms[key].params

# alignment_tforms[key]._apply_mat()
# alignment_tforms[key].inverse()

# alignment_tforms[key].estimate()
# alignment_tforms[key].residuals()


np.linalg.inv(tf.params) == tf._inv_matrix
tf._apply_mat(tf._apply_mat(np.array([1,1]), tf.params), tf._inv_matrix) == np.array([[ 1.,  1.]])
