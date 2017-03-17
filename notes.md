Remember
--------
- Devil is in the detail, truely...  
- A PhD student's estimation of the time required to finish a task has a standard deviation equal to estimation value.  

Maps to Use and Target Transfromations
--------------------------------------
- E5 floor plan
  - E5_01 [only two rooms] - (scale=(1.2,1.2), rotation=np.pi/2+0.04, translation=(1526,15))
  - E5_02 [no rooms]
  - E5_03 [no rooms - almost the same as E5_02]
  - E5_04 [BROKEN]
  - E5_05 [OK]
  - E5_06 [OK]
  - E5_07 [OK]
  - E5_08 [OK]

- F5 floor plan
  - F5_01 [two conference rooms, kinda broken]
  - F5_02 [just hall, not good]
  - F5_03 [two conference rooms]
  - F5_04 [two conference rooms and one office]
  - F5_05 [few rooms, but kinda broken]


TD
--

- [ ] over-decomposition's remedy; switch from edge-pruning to face growing
  for every neighboruing faces, merge if:
  1. faces are neighboring with same place categies
  2. the mutual edges are in the list `edges_to_prune`
  3. the emerging face must have the same shape as initial faces
  To do so, you need two methods:
  - [x] `shape = get_shape_descriptor(face)` - do not update attrubutes
  - [x] `new_face = merge_faces_on_fly(face1, face2)` - do not mutate arrangement

	OK! it tured out it is way more challenging and here is why:
	Face growing is supposed to consider face shapes!
	This means that the process need to be iterative, updating the arrangement
	with merged faces. this requires runing:
		- arrange._decompose()
		- edge/node occupancy
	And these guys are quite expensive!
	I can't afford to run them in every iteration that could go one for a while!

- [ ] `edge_pruning` with place categories is suseptible to missing edges if a signle room has two categories__
		remedy this with either:
			- playing with the threshold of face similarity (`are_same_category`)
			- put region seperating edges in a separate list other than `forbidden-list` (e.g. `category_baoundary`)
			  set the `low_occ_percent` very high (0.5) for normal edge (aggressive pruning)
			  set the `low_occ_percent` very high (0.01) for edges in category_baoundary (cautious pruning)

	* the followings mess-up prunning:
	  - [x] `are_same_category(face1,face2)` (debaugged)
	  - rule 3: `not_high_node_occupancy` - not more than `high_occ_percent`

- [ ] for hyp refute, use "OGM+skiz" or "distance image" and
	- [$R^2$](https://en.wikipedia.org/wiki/Coefficient_of_determination)
	- [MSE](https://en.wikipedia.org/wiki/Mean_squared_error)
	- [RSE](https://en.wikipedia.org/wiki/Residual_sum_of_squares)

- [ ] when done with hyp generation, use CPD and there is no need to adopt Siavash's code.
  Just find the transformation between `src` and `src_warp`


- [ ] big problem with cluster representation (average is not good)  
maybe include each member of the cluster as a hyp? not too many?

- [ ] proper setting for dbscan clustering

- [ ] convert `con_map` to `topo_map`:
		contract edges between two nodes with same place category label
		q: how about the coordinate of the new nodes?


- [ ] Show the local minima:
		For scale in [1,2,3]
			For rotation in [0, pi/2, pi, 3pi/2]
				Plot the surface (image alignment cost)

- [ ] `profile_nodes` only considers the `face.attributes['label_vote']`. How to convert to `face.attributes['label_count']`

- [ ] edge pruning based on point-based, not the new fatty version.

- [ ] pathes of the faces have the extent method:
		Returns the extents (*xmin*, *ymin*, *xmax*, *ymax*) of the path.
		Unlike computing the extents on the *vertices* alone, this
		algorithm will take into account the curves and deal with
		control points appropriately.
	  so finding the range for MBB neighborhood of circle face is not hard
	  just fix the damn DPE and you can support circles here too.

- [x] reject a transform if the maps do not overlap after warp.
  I did it, and many many transformations are rejected!
  But why are they being created in the first place?

- [x] Finally! one set of results that make sense!
  It came up when I switched tform estimation from 'similarity' to 'affine' and rejecting tforms with mismatching scales.

```
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}
prun_image_occupancy_thr = 200
prun_edge_neighborhood = 5
prun_node_neighborhood = 5
prun_low_occ_percent = .025 # below "low_occ_percent"
prun_high_occ_percent = .1 # not more than "high_occ_percent"
prun_consider_categories = True

con_map_neighborhood = 3
con_map_cross_thr = 9

hyp_gen_face_similarity = ['vote','count',None][2]
hyp_gen_tform_type = ['similarity','affine'][1]
hyp_gen_enforce_match = False

np.abs(tf.scale[0]-tf.scale[1])/np.min(tf.scale) < .1
# No need to reject weird transforms


['E5_layout', 'E5_07_tango']
min_s = int( .1* np.min([ len(arrangements[key].decomposition.faces)
                          for key in keys ]) )
cls = sklearn.cluster.DBSCAN(eps=0.0751, min_samples=min_s)
# this won't work with eps=0.051 - but does work with 0.06751

['E5_layout', 'E5_06_tango']
min_s = int( .1* np.min([ len(arrangements[key].decomposition.faces)
                          for key in keys ]) )
cls = sklearn.cluster.DBSCAN(eps=0.051, min_samples=min_s)
# also works with eps=0.0751, but a bit off - also a bit off with 0.06751

['E5_layout', 'E5_05_tango']
min_s = int( .1* np.min([ len(arrangements[key].decomposition.faces)
                          for key in keys ]) )
cls = sklearn.cluster.DBSCAN(eps=0.051, min_samples=min_s)
# also works with eps=0.0751, but a bit off

['E5_layout', 'E5_04_tango']
# It's a broken map and It has problem with abstraction

['E5_layout', 'E5_03_tango'] - almost identical - ['E5_layout', 'E5_02_tango']
# It has problem with abstraction, not enough generic-shape faces to match

['E5_layout', 'E5_01_tango']
# doesn't work!
```

- [x] I got one close answer with ( OOPs! it was false alarm! I used tf._inv_matrix!!! :'(( )

- [x] assume a rectangle with different hieght and width, the transformation estimated as 'similarity' with return two wrong estimation. If instead I use 'affine', then I can reject those by comparing `np.abs(tf.scale[0]-tf.scale[1])/np.min(tf.scale) <.1 or <.05`.  
	```
	src = np.array([ [0,0], [1,0], [1,1], [0,1] ])
	dst = np.array([ [3,1], [5,3], [4,4], [2,2] ])
	tf = skimage.transform.estimate_transform( 'affine', src, dst )
	```

- [x] Here is the proof that I have to use tf.params as forward transformation:  
```
src = np.array([ [0,0], [1,0], [1,1], [0,1] ])
dst = np.array([ [3,1], [4,2], [3,3], [2,2] ])
tf = skimage.transform.estimate_transform( 'similarity', src, dst )
src_warp = np.dot(tf.params, np.hstack( (src, np.array([[1],[1],[1],[1]]))).T ).T #[:,:-1]
fig, axes = plt.subplots(1,1, figsize=(20,12))
axes.plot( src[:, 0], src[:, 1], 'r.',label='source')
axes.plot( dst[:, 0], dst[:, 1], 'b.',label='destination')
axes.plot( src_warp[:, 0], src_warp[:, 1], 'g*',label='source transformed')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

- [x] should I discard face with label -1 in tform generation?
  yes, because pruning does not effect the occupied regions
  also, discard empty faces ( all attributes['label_count']==0 )

- [x] feature Rescaling instead of standardization!
  NO distribution of parameters (except for rotation) are actually normally distributed.
  Also, scale is a bit scewed, but nothing to worry about.  
  
```python
'''feature scaling (Rescaling)'''
min_ = np.min( parameters, axis=0 )
max_ = np.max( parameters, axis=0 )
parameters = (parameters - min_ ) / (max_-min_)
```	

- [x] use count as face label  

```python
mapali.histogram_of_face_category_distances(arrangements[key])
mapali.are_same_category(face1,face2, label_associations=None, thr=.4)
mapali.face_category_distance(face1,face2, label_associations)
mapali.construct_transformation_population(arrangements,
		connectivity_maps,
		similarity='vote') #similarity='count'
```

- [x] edge pruning's job is to convert "decomposition" of the arrangement to region segmentation. That is a room must become a single face after edge pruning. To do so, first do place categorization, and then make the occupancy_percent threshold dependant on that.  
	* If two regions have different categories, no edge pruning.
	* If they have the same label, the threshold could be high enough to allow fair amont of noise from the ending points of the segments that their side is located on a wall, like edges that slipt a room in two that might have high occupancy, but that shouldn't matter. The purpose of the percent threshold is to avoid the removal of edged that containt a doorway
such edges must have, supposedly high occupancy.

so first node prunning. Then edge prunning, considering labels of neighboring faces, and then prunning aggressively with a high threshold 

```python
mapali.plot_node_edge_occupancy_statistics(arrangements[key])
```



Data Flow
---------
* loading and constructions
  - `image[constant].load(file)`
  - `label_image[constant].load(file)`
  - `skiz[constant].load(file)`
  - `traits[constant].load(file)`
  - `arrangement[mutable].construct(traits)`
  - `connectivity_map[mutable].construct(arrangement.edge_crossing)`

* mutations and update
  - `(arrangement.face_label).update(label_image)`
  - `(arrangement.edge_node_occupancy).update(image)`
  - `(arrangement.edge_crossing).update(skiz)`
  - `(arrangement.gaph).mutate( arrangement.edge_node_occupancy, optional:arrangement.face_label)`
	
* dependencies
  - `(arrangement.face_label).depends(arrangment.graph)`
  - `(arrangement.edge_crossing).depends(arrangment.graph)`
  - `(connectivity_map).depends(arrangment.graph, arrangement.edge_crossing)`


Alignment, approach outline
---------------------------
In general there are two approaches for alignment: optimization and search.
	* Search essentially returns an association between elements of the two signal, hence requires an instanciation of the signals into abstract entities. The association would be between those entities. Depending on the level of abstraction (instanciation) the search space could be small or very big. In a big search space, if a well-discriminating descriptor is not available for the entities, one has to perform a full search on the search space, which often is intractable.
	* Optimization on the other hand is prone to local minima, and for a challenging search space requires a reliable initial guess to converge to the desired solution.


phase one: Search, finding local minima
phase two: optimization, optimizing within the vicinity of the local minima




transformation parameters standarzation for clustering
------------------------------------------------------
It is important to standardize, because the translation values are dominantly larger than scale and rotation values.
On the other hand, enough of samples are far off to mis-guide the std and mess the standardizatio. due to this problem, any sample that has translation or scale beyond their respective threshold are dropped out of the pool.



Experiments and results
-----------------------
* Schewrtfeger - couldn't compile!
* Daniels image alignment -> all fall into local minima
* CPD -> all fall into local minima

* Randomized hyps followed by (CPD, Image alignment, line-segment match)
* duality and place category based hyps followed by (CPD, Image alignment, line-segment match)


graph-based association
-----------------------
centrality measures (eigen-values, load, harmonic,...) and distance measures (centre, periphery, eccentricity, diameter,...) are sensitive to partiallity of the maps. That is the centre of two connectivity maps of the same environment, but partially overlapping, would have their centers at different locations.  
On the other hand, isomorphism is sensitive to levels of abstraction in different modalities as well as the repeating patterns of the environment.  
Among the two approaches, However, the isomorphism is more applicable since it can handle the repeating patterns by enumerating all possible matches, and tackle the discripency in abstraction by employing minor graphs. Nevertheless, it remains a challenging problem since it would depend on multiple factors and has to deal with two problems of different natures.  
This is a good point to introduce additional sources of information to narrow the search.
These additional informations could be the congruence constraint over the transformation implied by the associations, and the category cues from shape the of the environments.


Brokenness
----------
if a map is broken, how to recognize the brokenness? If possible to decompose the map into consistent local maps, then they could be separately aligned with the prior CAD map.

Note
----
* Target is arrangement, hence yaml
* Svg -> direct parsing -> yaml
* Bitmap -> GUI -> yaml
* Point cloud -> GUI (ori-dom?) -> yaml


Meeting with Martin
-------------------
- new topic: multi-modal maps (ogm/layout/pointcloud) alignment and fusion
--- load ply into annotation gui (how to detect dom_ori?)

* giving up duality? Not actually, it's just there is a better way for the prime mode, than shape matching. The difference in abstraction is very challenging, and The global consistent assume means it's better to find a similarly from nodes to nodes. The shape matching in the end will use the same source of information anyway, just with more detailed search. It might be better for the broken maps, but that is not the assumption now. So stick to a global match between nodes.
* the problem of abstraction doesn't interfere with node matching, there might be extra nodes, but many will supposedly coincide.
* for the node (or even the prime mode) there's no need to look at the occupancy map.
* but what about the topology? The topological graph certainly doesn't have the global geometric consistency. Over decomposition would change the geometric location and consistency of the dual graphs.

subgraph isomorphism 
--------------------
sub-graph isomorphism based on the connectivity graph

these seems like good starting point:
	- [Graph Isomorphism](https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.isomorphism.html)
	- [isomorphism.vf2](http://networkx.readthedocs.io/en/stable/reference/algorithms.isomorphism.vf2.html)
	- [core](http://networkx.readthedocs.io/en/stable/reference/algorithms.core.html)
	- [assortativity](http://networkx.readthedocs.io/en/stable/reference/algorithms.assortativity.html)
	- [vf2 match helpers](http://networkx.readthedocs.io/en/stable/reference/algorithms.isomorphism.vf2.html#match-helpers)	
	- [distance measures](http://networkx.readthedocs.io/en/stable/reference/algorithms.distance_measures.html)
		- The center of a graph G is the set of vertices of graph eccentricity equal to the graph radius (i.e., the set of central points).
		- The length max_(u,v)d(u,v) of the "longest shortest path" (i.e., the longest graph geodesic) between any two graph vertices (u,v) of a graph, where d(u,v) is a graph distance. 
		- The eccentricity epsilon(v) of a graph vertex v in a connected graph G is the maximum graph distance between v and any other vertex u of G. For a disconnected graph, all vertices are defined to have infinite eccentricity (West 2000, p. 71). 
		- The periphery of a graph G is the subgraph of G induced by vertices that have graph eccentricities equal to the graph diameter.
		- The radius of a graph is the minimum graph eccentricity of any graph vertex in a graph. A disconnected graph therefore has infinite radius (West 2000, p. 71). 


```python
# an toy-example for trying out the api to different algorithms 
nodes = [ [idx, {}] for idx in [0,1,2,3,4] ]
edges = [ (0,1), (0,2), (0,3), (0,4)]

G1 = nx.MultiGraph()
nodes_extra = [ [idx, {}] for idx in [5,6] ]
edges_extra = [ (4,5), (5,6) ]
G1.add_nodes_from( nodes+nodes_extra )
G1.add_edges_from( edges+edges_extra )

G2 = nx.MultiGraph()
nodes_extra = [ [idx, {}] for idx in [] ]
edges_extra = [ (1,2) ]
G2.add_nodes_from( nodes+nodes_extra )
G2.add_edges_from( edges+edges_extra )

GM = nx.algorithms.isomorphism.GraphMatcher(G1,G2)
print ( GM.is_isomorphic() ) # Returns True if G1 and G2 are isomorphic graphs.
print ( GM.subgraph_is_isomorphic() ) # Returns True if a subgraph of G1 is isomorphic to G2.
```

for subgraph-isomorphism, I have to generate subgraphs and compare
d = [] - degree: finds the least appearing node-degrees that occure in both graphs
n = [] - neighborhood - n>1 - for n=1, all subgraphs with same degree are isomorphic (2<=n<=4)
for each graph create subgraphs sg_nd for all n and d
between subgraphs of the two arrangements with samge n and d, find isomorphism -> gen_hyp
gen_hyp: node match in dual graph means face match in prime graph, generate all and find concensus

```python
# generate subgraphs
nodes_degrees = connectivity_map.degree()
degrees = {nodes_degrees[key] for key in nodes_degrees}
nx.subgraph(G, nbunch) # induce subgraph of G on nodes in nbunch
```

```python
for key in keys:
    print ('---------- {:s} ----------'.format(key) )
    print ( 'center: ', nx.center(con_map_max_subgraph[key]) )
    print ( 'diameter: ', nx.diameter(con_map_max_subgraph[key]) )
    print ( 'eccentricity: ', nx.eccentricity(con_map_max_subgraph[key]) )
    print ( 'periphery: ', nx.periphery(con_map_max_subgraph[key]) )
    print ( 'radius: ', nx.radius(con_map_max_subgraph[key]) )
```

USING LINALG (for solving the association problem)
--------------------------------------------------
I have a feature vector for each map that describes its characterisitcs in topological space (based on the connectivity graph, e.g. node degrees), and geometric space (based on arrangements and place categories).  
The first question is whether it is possible to find a unique and robust to noise feature vector for describing each map.  
The problem of association is that I don't know which feature dimensions of the first map corresponds to the dimensions of the second map. Most importantly, which place category of the first map corresponds to which category of the second map. I might be able to find some heuristic cues to find association between node degrees.  
Now let's see if I can find a way to use concepts of linear algebra (e.g. determinant, inverse matrix(i.e. solution to the system of linear equations), eigenvectors, etc ) to solve the association problem. That is to find a transformation that maps the feature vector of one map to the feature vector of the second map.  

so there could be a dimension for every node-degree from 1 to max(node-degree)  
skip zero as it represents orphan faces which are probably non-relevant.  
include all degrees, even if there is no node with such a degree in either maps.  
It might sound convenient to keep a one-2-one correspondance between faces with same node-degrees, but it ain't right. remember two corresponding faces might have different node-degrees.  

Maybe I should describe each face with a vector? node-degree, place-label, etc  
and find a mapping that associates faces from one map to another.  
this mapping will probably change the node-degree and place category for most faces, but not nessecarily all faces.  
so maybe the mapping-transformation is not linear after all.  
Anyhow, such mapping-transformation (i.e. face-2-face association) could be interpreted as a rigid transforamtion.  
The idea is not find those mapping-transformation, which have the most coherent corresponding rigid transformation.  

The feature vectors of the maps could be measures based on their interpretations plus the applied transformations. These vectors could provide a measure to check the validity of the solution (i.e. transformations). The transformation might need to be a compound of multiple transfromation over each map to meet each other at a certain point. After transformations, the feature of the maps should look alike.  

Using place categories, I am becoming dependant on the scale, inheritingly from features.  

Given that I described faces with vectors in a features space, I can use SVD and eigenvalues/vectors to estimate the transformation, I also have a rough association between data points or vectors in the feature space, and also a geometric coherency to constraint the search for the solution...  

If I manage to properly construct that feature space, then I have to find a transformation that converts map 1 to map 2. The eigenvector of that transformation would be in direction of those dominant features that stay the same for both maps. The problem is twofold, one to construct the feature space, and secondly to estimate a transformation between maps in absence of association. The second crux makes the problem circular, IE for this approach I need association, and if I had association I wouldn't need this approach.  

What if the transformation is in the form of abstraction-adjustment and pattern matching? then the problem of association is addressed meanwhile. But back to crux one, what should be the feature space that supports adjustment and matching? The matching problem should be substituted with distance metric, and find the answer to the substitution.  

* listen again to the eigenvector description [here](https://soundcloud.com/edwardoneill/steven-strogatz-on-teaching-eigenvectors-and-eigenvalues)
* watch again the [chp.6](https://www.youtube.com/watch?v=uQhTuRlWMxw) and [chp.10](https://www.youtube.com/watch?v=PFDu9oVAE-g) of linear algebra series by 3blue1brown
* [ PageRank Algorithm - The Mathematics of Google Search](http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html)  
* [The eigenvector of "Why we moved from language X to language Y"](https://erikbern.com/2017/03/15/the-eigenvector-of-why-we-moved-from-language-x-to-language-y.html)


code snippet dumpster
---------------------

```python
# visualizing forbidden edges and edges in between faces with different categories

forbidden_edges  = arrange.get_boundary_halfedges()
diff_edges = []
for (f1Idx, f2Idx) in itertools.combinations( range(len(arrange.decomposition.faces)), 2):
    face1 = arrange.decomposition.faces[f1Idx]
    face2 = arrange.decomposition.faces[f2Idx]
    if ( face1.attributes['label_vote'] != face2.attributes['label_vote']):
        diff_edges.extend( arrange.decomposition.find_mutual_halfEdges(f1Idx, f2Idx) )

aplt.plot_edges(axes[idx], arrange, halfEdgeIdx=diff_edges)

mapali.set_edge_occupancy(arrange, image,
                          occupancy_thr=200,
                          neighborhood=5,
                          attribute_key='occupancy')

for s,e,k in arrange.graph.edges(keys=True):
    if ((s,e)==(43, 54)) or ((s,e)==(54, 43)):
        o, n = arrange.graph[s][e][k]['obj'].attributes['occupancy']
        print( (s,e,k), (o, n, float(o)/n), ((s,e,k) in forbidden_edges), ((s,e,k) in diff_edges) )
```

```python
# example of `merge_faces_on_fly`
for f1_idx in range(len(arrange.decomposition.faces)):
    for f2_idx in arrange.decomposition.find_neighbours(f1_idx):
        new_face = utls.merge_faces_on_fly(arrange, f1_idx, f2_idx)
        if new_face is None:
            print (f1_idx, f2_idx)


# This plotting doesn't work and I don't have time to fix it
import arrangement.plotting as aplt
import matplotlib.patches as mpatches
fig, axes = plt.subplots(1,1, figsize=(20,12))
mapali.plot_arrangement(axes, arrange, printLabels=False)

for f1_idx in [0]: #range(len(arrange.decomposition.faces)):
    for f2_idx in arrange.decomposition.find_neighbours(f1_idx):
        new_face = utls.merge_faces_on_fly(arrange, f1_idx, f2_idx)
        
        if new_face is not None:
            patch = mpatches.PathPatch(new_face.get_punched_path(),
                                       facecolor='g', edgecolor=None)
            p = axes.add_patch(patch)
            time.sleep(0.5)
            p.remove()
            
        elif new_face is None:
            p = []
            f1 = arrange.decomposition.faces[f1_idx]
            f2 = arrange.decomposition.faces[f2_idx]
            patch = mpatches.PathPatch(f1.get_punched_path(),
                                       facecolor='r', edgecolor=None)
            p.append( axes.add_patch(patch) )
            patch = mpatches.PathPatch(f2.get_punched_path(),
                                       facecolor='r', edgecolor=None)
            p.append( axes.add_patch(patch) )
            
            time.sleep(0.1)
            for p_i in p:
                p_i.remove()

plt.axis('equal')
plt.tight_layout()
plt.show()
```

```python
# an example for manual transformation (keys = ['E5_layout', 'E5_07_tango'])
src = np.array([ [346,907], [518,907], [518,994], [346,994] ])
dst = np.array([ [529,498], [538,285], [652,290], [643,503] ])
tform = skimage.transform.estimate_transform( 'similarity', src, dst )
mapali.plot_trnsfromed_images(images[keys[0]], images[keys[1]], tformM=tform.params)
```

```python
for lbl in unique_labels:
    if lbl != -1:
        class_member_idx = np.nonzero(labels == lbl)[0]
        class_member = [ tforms[idx]
                         for idx in class_member_idx ]
        print ('*********************** label {:d}'.format(lbl))

        print ('\t*********************** translations :')
        for tf in class_member:
            print (tf.translation)

        print ('\t*********************** scale :')
        for tf in class_member:
            print (tf.scale)

        print ('\t*********************** rotation :')
        for tf in class_member:
            print (tf.rotation)
```

```python
# feature scaling (standardization)
parameters -= np.mean( parameters, axis=0 )
parameters /= np.std( parameters, axis=0 )
```

```python
# using unit vector transformation to represent alignments
U = np.array([1,1,1])
transformed = np.stack([ np.dot(tf.params, U)[:2]
                         for tf in tforms ], axis=0)
```

```python
# 2d transformation  feature plotting
skp = 1
fig, axes = plt.subplots(1,1, figsize=(12,6))
axes.plot(transformed[:,0][::skp], transformed[:,1][::skp], 'r.', alpha = .3)
axes.plot(parameters[:,0][::skp], parameters[:,1][::skp], 'r.', alpha = .3)
axes.plot(parameters[:,2][::skp], parameters[:,3][::skp], 'r.', alpha = .3)
plt.axis('equal')
plt.tight_layout()
plt.show()
```

```python
# 3D - mayavi - transformation feature plotting
# http://docs.enthought.com/mayavi/mayavi/index.html
# sudo pip install mayavi

from mayavi import mlab
mlab.points3d(parameters[:,0][::skp],
              parameters[:,1][::skp],
              parameters[:,2][::skp],
              scale_factor=.25)
mlab.show()
```

```python
nx.periphery(con_map_max_subgraph[key])
nx.center(con_map_max_subgraph[key])
```

```python
import collections
con_map_max_subgraph ={}
most_freq_label = {}
largest_face = {}
highest_deg_node = {}
```

```python
for key in keys:
    # removing nodes from connectivity map, if the face has label -1
    faces = arrangements[key].decomposition.faces
    nodes_to_remove = [ f_idx
                        for f_idx in connectivity_maps[key].node.keys()
                        if faces[f_idx].attributes['label'] == -1 ]
    connectivity_maps[key].remove_nodes_from(nodes_to_remove)
    
    # find biggest connected_component_subgraphs for every connectivity
    subGraphs = list(nx.connected_component_subgraphs( connectivity_maps[key] ) )
    no_node = [len(sg.node) for sg in subGraphs]    
    idx, val = max(enumerate(no_node), key=operator.itemgetter(1))        
    con_map_max_subgraph[key] = subGraphs[idx]

    # most_freq_label
    labels_lst = [face.attributes['label']
                  for face in arrangements[key].decomposition.faces]
    most_freq_label[key] = max(set(labels_lst), key=labels_lst.count)

    # face area
    faces_area = [ face.get_area()
                   for face in arrangements[key].decomposition.faces ]
    index, value = max(enumerate(faces_area), key=operator.itemgetter(1))
    largest_face[key] = index

	#  highest degree nodes
    nd = connectivity_maps[key].degree()
    nd_lst = [nd[k] for k in nd.keys() if nd[k]!=0]
    print ( collections.Counter(nd_lst) )
    idx, val = max( [ (k, nd[k]) for k in nd.keys() ],
                    key=operator.itemgetter(1))
    highest_deg_node[key] = idx
```

```python
# plotting - marking specific faces

# mark largest face
p  = arrangements[key].decomposition.faces[largest_face[key]].attributes['centre']
axes[idx].plot (float(p[0]), float(p[1]), 'b*', markersize=15)
	
# mark highest degree node
p = connectivity_maps[key].node[highest_deg_node[key]]['coordinate']
axes[idx].plot (float(p[0]), float(p[1]), 'b*', markersize=15)

# mark "most frequent labels"
for face in arrangements[key].decomposition.faces:
	if face.attributes['label'] == most_freq_label[key]:
		p  = face.attributes['centre']
		axes[idx].plot (float(p[0]), float(p[1]), 'r+', markersize=15)

# mark faces with e-measure higher than average
for n_idx,e_val in enumerate(e):
    if e_val > e.mean():
        f_idx = connectivity_maps[key].node.keys()[n_idx]
        p  = arrangements[key].decomposition.faces[f_idx].attributes['centre']
        axes[idx].plot (float(p[0]), float(p[1]), 'r+', markersize=15)

# mark centre of the connectivity map
for f_idx in nx.center(con_map_max_subgraph[key]):
    p  = arrangements[key].decomposition.faces[f_idx].attributes['centre']
    axes[idx].plot (float(p[0]), float(p[1]), 'b*', markersize=15)

# mark periphery of the connectivity map
for n_idx in nx.periphery(con_map_max_subgraph[key]):
    p  = arrangements[key].decomposition.faces[f_idx].attributes['centre']
    axes[idx].plot (float(p[0]), float(p[1]), 'r+', markersize=15)
```
