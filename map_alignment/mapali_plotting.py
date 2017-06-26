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

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms

import arrangement.plotting as aplt

################################################################################

def plot_connectivity_map(axes, connectivity_map, clr='g', alpha=.7):
    '''
    nodes of connectivity map must contain 'coordinate' key and corresponding value   
    '''
    X,Y = zip( *[ connectivity_map.node[key]['coordinate']
                  for key in connectivity_map.node.keys() ] )
    axes.plot(X,Y, clr+'o', alpha=alpha)
    for (s,e,k) in connectivity_map.edges(keys=True):
        X,Y = zip( *[ connectivity_map.node[key]['coordinate']
                      for key in (s,e) ] )
        axes.plot(X,Y, clr+'-', alpha=alpha)
    
    return axes

################################################################################
def plot_arrangement(axes, arrange, printLabels=False ):
    '''
    '''
    aplt.plot_edges (axes, arrange, alp=.3, col='b', printLabels=printLabels)
    aplt.plot_nodes (axes, arrange, alp=.5, col='r', printLabels=printLabels)
    return axes

################################################################################
def plot_text_edge_occupancy(axes, arrange, attribute_key=['occupancy']):
    '''
    '''
    for s,e,k in arrange.graph.edges(keys=True):
        p1 = arrange.graph.node[s]['obj'].point
        p2 = arrange.graph.node[e]['obj'].point
        x, y = p1.x.evalf() , p1.y.evalf()
        dx, dy = p2.x.evalf()-x, p2.y.evalf()-y

        txt = ''
        for att_key in attribute_key:
            o,n = arrange.graph[s][e][k]['obj'].attributes[att_key]
            txt += '({:.4f})'.format(float(o)/n)
        
        axes.text( x+(dx/2), y+(dy/2), 
                   txt, fontdict={'color':'k',  'size': 10})

################################################################################
def plot_place_categories (axes, arrangement, alpha=.5):
    '''
    '''
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
    '''
    '''
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
    '''
    '''
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

    X: destination
    Y: source
    '''

    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue')
    ax.axis('equal')
    plt.draw()
    plt.pause(0.01**5)

################################################################################
def plot_transformed_images(images, keys,
                            tformM=None,
                            axes=None,
                            title=None,
                            pts_to_draw=None,
                            save_to_file=False):
    '''
    src,dst (images 2darray)

    tformM (2darray 3x3 - optional)
    default (None) will use an identity matrix

    axes (matplotlib axes - optional)
    if axes is None, the method will plot everythnig self-contained
    otherwise, will plot over provided axes and will return axes

    title (string - optional)
    title for axes

    pts_to_draw (dictionary - optional)
    pts_to_draw['pts'] contains a point set to plot
    pts_to_draw['mrk'] is the marker (eg. 'r,', 'b.') for point plot
    if pts_to_draw['mrk'] is not provided, marker is set to 'r,'

    '''

    src, dst = images[keys[0]], images[keys[1]]

    aff2d = matplotlib.transforms.Affine2D( tformM )
    # aff2d._mtx == tformM

    return_axes = True
    if axes is None:
        return_axes = False
        fig, axes = plt.subplots(1,1, figsize=(20,12))

    # drawing images and transforming src image
    im_dst = axes.imshow(dst, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    im_src = axes.imshow(src, origin='lower', cmap='gray', alpha=.5, clip_on=True)
    im_src.set_transform( aff2d + axes.transData )

    # finding the extent of of dst and transformed src
    xmin_d,xmax_d, ymin_d,ymax_d = im_dst.get_extent()
    x1, x2, y1, y2 = im_src.get_extent()
    pts = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    pts_tfrom = aff2d.transform(pts)    

    xmin_s, xmax_s = np.min(pts_tfrom[:,0]), np.max(pts_tfrom[:,0]) 
    ymin_s, ymax_s = np.min(pts_tfrom[:,1]), np.max(pts_tfrom[:,1])

    # setting the limits of axis to the extents of images
    axes.set_xlim( min(xmin_s,xmin_d), max(xmax_s,xmax_d) )
    axes.set_ylim( min(ymin_s,ymin_d), max(ymax_s,ymax_d) )

    if pts_to_draw is not None:
        pts = pts_to_draw['pts']
        mrk = pts_to_draw['mrk'] if 'mrk' in pts_to_draw else 'r,'
        axes.plot(pts[:,0], pts[:,1], mrk)

    # # turn off tickes
    # axes.set_xticks([])
    # axes.set_yticks([])

    if title is not None: axes.set_title(title)


    if return_axes:
        return axes
    else:
        if save_to_file:
            plt.savefig('_'.join(keys)+'.png', bbox_inches='tight')
            plt.close(fig)
        else:
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


################################################################################
def plot_face2face_association_match_score(arrange_src, arrange_dst,
                                           f2f_association, f2f_match_score):
    '''
    '''
    ### fetch the center of faces
    face_cen_src = np.array([face.attributes['centre']
                             for face in arrange_src.decomposition.faces])
    face_cen_dst = np.array([face.attributes['centre']
                             for face in arrange_dst.decomposition.faces])

    fig, axes = plt.subplots(1, 1, figsize=(20,12))

    ### source : blue
    aplt.plot_edges (axes, arrange_src, alp=.3, col='b', printLabels=False)
    axes.plot(face_cen_src[:,0], face_cen_src[:,1], 'b*', alpha=.5)

    ### destination : green
    aplt.plot_edges (axes, arrange_dst, alp=.3, col='g', printLabels=False)
    axes.plot(face_cen_dst[:,0], face_cen_dst[:,1], 'g*', alpha=.5)

    ### face to face association and match score
    for src_idx in f2f_association.keys():
        dst_idx = f2f_association[src_idx]
        x1,y1 = face_cen_src[ src_idx ]
        x2,y2 = face_cen_dst[ dst_idx ]
        # plot association
        axes.plot([x1,x2], [y1,y2], 'r', alpha=.5)
        # print match score
        score = f2f_match_score[(src_idx,dst_idx)]
        axes.text(np.mean([x1,x2]) , np.mean([y1,y2]),
                  str(score), fontdict={'color':'k',  'size': 10})

    axes.axis('equal')
    plt.tight_layout()
    plt.show()
