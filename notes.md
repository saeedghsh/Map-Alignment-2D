### load data, and perform interpretation 
deploying arrangement
arrangement pruning
place categories to faces assignment
setting ombb attribute of faces
connectivity map construction and node profiling
 

### generate hypothesis
label_associations = mapali.label_association(arrangements, connectivity_maps)
tforms.extend (mapali.align_ombb(face_src,face_dst, tform_type='affine'))
tforms = mapali.reject_implausible_transformations( tforms )

### pick winning hypothesis
if tforms.shape[0] < 100:
    pick best tform accourding to arrangement_match_score
    hypothesis = tforms[best_idx]
else:
	cluster tforms into groups and find a representative for each cluster
    find the best cluster accourding to arrangement_match_score
	find the best element in the cluster accourding to arrangement_match_score

### optimize the final solution






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
  - E5_09 [OK]
  - E5_10 [OK]
  
- F5 floor plan
  - F5_01 [two conference rooms, kinda broken]
  - F5_02 [just hall, not good]
  - F5_03 [two conference rooms]
  - F5_04 [two conference rooms and one office]
  - F5_05 [few rooms, but kinda broken]
  - F5_06 [seems ok!]


Configurations:
---------------
for my_home:
```
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}

prun_image_occupancy_thr = 200
prun_edge_neighborhood = 5
prun_node_neighborhood = 5
prun_low_occ_percent = .05 # below "low_occ_percent" # for home
prun_high_occ_percent = .1 # not more than "high_occ_percent"
prun_consider_categories = [True, False][1]

# don't do face-growing 
face_low_occ_percent = .05 # self below "low_occ_percent"
face_high_occ_percent = .1 # no nodes more than "high_occ_percent"
face_consider_categories = [True, False][0]
face_similar_thr = 0.4

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

hypgen_face_similarity = ['vote','count',None][2] -> actually I do use 'vote'
hypgen_tform_type = ['similarity','affine'][1]
hypgen_enforce_match = False


>>> find label_associations and align ombb of same category faces
>>> reject bad tforms -> At this point I often end-up with low number of tforms (<30)
>>> arrangement_match_score_fast() is capable of picking the winner
```


Drawbaxk and limitations
------------------------
- computation cost - mention bottle-necks and ideas how to lower the cost.
- global consistency requirement.
- only works if the abstraction in inputs meet on a comparable levels. That is to say, if one region is represented with $n$ faces (ideally one) with a particular configuration, the cooresponding region in the other map should be also represented with same number of faces and similar configuration. However, this assumption is not as restrictive as it sounds. Firstly, it should be noted that not all regions need to match in the input maps, but enough to generate plausible hypotheses. Secondly, relying on the assumption of well structure environment it is expected for different maps of the same environment to result in similar deconposition.


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
	- l2
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

- [ ] detection of brockenness in maps
	- can not rely on arrangement since it relies on the assumption of global consistency
	- should it be based on the occupancy map?

- [ ] Show the local minima:
		For scale in [1,2,3] - For rotation in [0, pi/2, pi, 3pi/2]
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


Region intersection and union
-----------------------------
So far I couldn't find a decent library (or any for that matter) to handle path intersection operation.
	* svgpathtools has an intersection function for paths, but it only returns intersection points
    * I couldn't find anything in inkex, but I didn't search good enough.
    * shapely does not support paths as matplotlib and svg does, it has only LinearRings of line segement.
    * matplotlib's intersects_path() only returns Boolean
So the current implementation pixelates faces and computes area accordingly


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


Image Registration and Alignment
--------------------------------
- [ICP](https://engineering.purdue.edu/kak/distICP/ICP-2.0.html)
- [(Astronomical) Image Registration](http://image-registration.readthedocs.io/en/latest/) - NOPE!
- [menpo](https://github.com/menpo/menpo) and [menpofit](https://github.com/menpo/menpofit) - very comprehensive, see [lucas-kanad](http://www.menpo.org/menpofit/lk.html) example for image alignment.
  but! it downgrades matplotlib to 1.5.
  there are lots of problem with it! the documentation does not match the implementation.
  see `~/Dropbox/myGits/dev/menpofit_test.py`


Coherent Point Drift
--------------------
python implementations of coherent point drift (in the order of being user-friendly):
- [Python-CPD](https://github.com/siavashk/Python-CPD)
- [Coherent-Point-Drift-Python](https://github.com/Hennrik/Coherent-Point-Drift-Python)
- [coherent-point-drift](https://github.com/kwohlfahrt/coherent-point-drift)
- [pycpd](https://github.com/dpfau/pycpd/blob/master/test/test_fgt.py)
The algorithm works in a way that the first gradual transformation will map the source point set to location at the center of the destination point set with a severe down scaling. this behaviour is apparantly due the algorithm, not the implementation. see the examples in the paper.
And after such a transformation it can not recover. This happens even if I provide the "corrrect' initial transformation. 


transformation parameters standarzation for clustering
------------------------------------------------------
It is important to standardize, because the translation values are dominantly larger than scale and rotation values.
On the other hand, enough of samples are far off to mis-guide the std and mess the standardizatio. due to this problem, any sample that has translation or scale beyond their respective threshold are dropped out of the pool.


Why Abstraction?
----------------
Upwards flow is about structuring the data into an abstract representation that is shared among different sensors, followed by assigning semantic labels to structured instances. What follows is a list of potentially target applications for such a process:
1. The zipper -> enriching the robot's representation of the world by merging multiple sources of information.
1.a. Semantically annotated instances of the maps can provide cues for finding the vicinity of the solution to the merging problem. While the complete search space might be intractable, such cues could narrow down the search space and simplifying the associated optimization problem.
1.b. the shared abstract representation make it possible to merge maps of different natures in the absence of a spatiotemporal transformation between the sensors.
2. Simplifying the path planning (and task assignment?) of agents in a crowded environment with the risk of collision.
3. Automatic surveying, resulting in a CAD drawings understandable by humans. This was the original objective, right?



Experiments and results
-----------------------
To justify the complexity of the solution, we refer the reader to the results of solving the problem by naive approaches where we treat the maps merely as bitmap images.

Our approach is to solve the alignment problem by finding local minima and perform optimization within the vicinity of those local minima.
We tackle this challenge by finding rough association between the maps which provides aforementioned local minima, and optimimiz the alignment with the conventional image/point-set registration.
Towards solving the association problem, we employ a combination of multiple interpretation which are described in chapter x.

* Schewrtfeger - couldn't compile!
* Daniels image alignment -> all fall into local minima
* CPD -> all fall into local minima
* radon-hough based can't solve for all (tx,ty,s,r)

* Randomized hyps followed by (CPD, Image alignment, line-segment match)
* duality and place category based hyps followed by (CPD, Image alignment, line-segment match)

Future Work - Centainty Metric
------------------------------
Complementing the framework with a data-friven reasoning batch, to verify/asses the alignment hypotheses on structural-level (arrangement alignment, i.e abstraction level) and low-level (sensory data).
This batch provides a centainty metric and potentially cues on hypotheses improvement.
could remove "global consistency" assumption to deliver solutions to broken maps?

Future Work - Fusion
--------------------
The ability to fuse information from different sources of inputs is a crucial step toward building richer models.
One way is to fuse based on references that is to say semantic similarities of instances in the model. Another would be to find similarity in the representation of the models if the different source of input could be represented with a mutual shared representation.
- Verification: Find a way to merge the maps without semantics, then compare the results. I think the selling point of semantics is to provide reasonable initial guesses. Emphasis the scale problem.
- Parallel to developing the outline of the fusion, prepare the "simulation", "ontology", and "inter-modal map association". check out the "knowledge representation formalisms" folder.
- Ontologies
- Architecture (Bayes?)


Brokenness
----------
if a map is broken, how to recognize the brokenness? If possible to decompose the map into consistent local maps, then they could be separately aligned with the prior CAD map.


Data-File Flow
--------------
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


graph-based association
-----------------------
centrality measures (eigen-values, load, harmonic,...) and distance measures (centre, periphery, eccentricity, diameter,...) are sensitive to partiallity of the maps. That is the centre of two connectivity maps of the same environment, but partially overlapping, would have their centers at different locations.  
On the other hand, isomorphism is sensitive to levels of abstraction in different modalities as well as the repeating patterns of the environment.  
Among the two approaches, However, the isomorphism is more applicable since it can handle the repeating patterns by enumerating all possible matches, and tackle the discripency in abstraction by employing minor graphs. Nevertheless, it remains a challenging problem since it would depend on multiple factors and has to deal with two problems of different natures.  
This is a good point to introduce additional sources of information to narrow the search.
These additional informations could be the congruence constraint over the transformation implied by the associations, and the category cues from shape the of the environments.

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


Minor issues and tips
---------------------
 - In the arrangement/demo.py `isinstance(arrange.decomposition, arr.Decomposition)` is True, but in the Map-Alignment-2D/runMe.py `isinstance(arrange.decomposition, arr.Decomposition)` is False.
   It resolves! Not sure how, but probably by reload(arr) in runMe.py. but persists!
	   is it because of the copy?
	   or is is because of modification in the instance?
   It happens when I reload(arr) (also reload(mapali) which has reload(arr) in it). 
   

code snippet dumpster
---------------------

```python
```

```python
```

```python
    # ###### internal plottig of skiz_bitmap method - for the debuging and fine-tuning
    # internal_plotting = False
    # if internal_plotting:

    #     import matplotlib.gridspec as gridspec
    #     gs = gridspec.GridSpec(2, 3)
        
    #     # image
    #     ax1 = plt.subplot(gs[0, 0])
    #     ax1.set_title('original')
    #     ax1.imshow(original, cmap = 'gray', interpolation='nearest', origin='lower')
        
    #     # image_binary
    #     ax2 = plt.subplot(gs[1, 0])
    #     ax2.set_title('image')
    #     ax2.imshow(image, cmap = 'gray', interpolation='nearest', origin='lower')
        
    #     # dis
    #     ax3 = plt.subplot(gs[0, 1])
    #     ax3.set_title('dis')
    #     ax3.imshow(dis, cmap = 'gray', interpolation='nearest', origin='lower')
        
    #     # grd_binary
    #     ax4 = plt.subplot(gs[1, 1])
    #     ax4.set_title('abs(grd) [binary_inv]')
    #     ax4.imshow(grd_abs, cmap = 'gray', interpolation='nearest', origin='lower')
    #     # ax4.imshow(grd_binary_inv, cmap = 'gray', interpolation='nearest', origin='lower')
        
    #     # voronoi
    #     ax5 = plt.subplot(gs[:, 2])
    #     ax5.set_title('skiz')
    #     ax5.imshow(skiz, cmap = 'gray', interpolation='nearest', origin='lower')

    # plt.show()

```

```python
### plotting src (transformed) and dst images for the "center" elemnt of the cluster 
for lbl in unique_labels:
    if lbl != -1:
        class_member_idx = np.nonzero(labels == lbl)[0]
        class_member = [ tforms[idx]
                         for idx in class_member_idx ]
        # it's actually not that bad! inter-cluster tranforms are pretty close
        # I once (just now) checked about 200 of them...!
        
        # pick the one that is closest to all others in the same group
        params = parameters[class_member_idx]
        dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(params, 'euclidean'))
        dist_arr = dist_mat.sum(axis=0)
        tf = class_member[ np.argmin(dist_arr) ]

        mapali.plot_transformed_images(images[keys[0]], images[keys[1]],
                                       tformM=tf.params,
                                       title='cluster {:d}'.format(lbl))
```

```python
for tf in tforms:
	mapali.plot_transformed_images(images[keys[0]], images[keys[1]], tformM= tf.params)
```

```python
### plot the histogram of edges' average distance value (from distance images)
col ={keys[0]:'r', keys[1]:'b'}
for key in keys:
    vals = []
    for (s,e,k) in arrangements[key].graph.edges(keys=True):
        neighbors_dis_val = arrangements[key].graph[s][e][k]['obj'].attributes['distances']
        sum_ = neighbors_dis_val.sum() / 255.
        siz_ = np.max([1,neighbors_dis_val.shape[0]])
        vals.append( float(sum_)/siz_ )
    plt.hist(vals, bins = 60, color=col[key], alpha=.7, label=key)
plt.legend()
plt.show()

```

```python

# prun_image_occupancy_thr = 200
# prun_edge_neighborhood = 5
# prun_node_neighborhood = 5
# prun_low_occ_percent = .025 # below "low_occ_percent" # for E floor
# prun_low_occ_percent = .15 # below "low_occ_percent" # for home
# # prun_low_occ_percent = .0125 # below "low_occ_percent"
# prun_high_occ_percent = .1 # not more than "high_occ_percent"
# prun_consider_categories = [True, False][1]


	######################################## edge pruning the arrangement
    print ('\t edge pruning arrangement wrt occupancy map ...') 
    arrange = mapali.prune_arrangement( arrange, image,
                                        image_occupancy_thr = prun_image_occupancy_thr,
                                        edge_neighborhood = prun_edge_neighborhood,
                                        node_neighborhood = prun_node_neighborhood,
                                        low_occ_percent  = prun_low_occ_percent,
                                        high_occ_percent = prun_high_occ_percent,
                                        consider_categories = prun_consider_categories)
```

```python
    ######################################## setting face attribute with shape description
    # for face matching and alignment - todo. make this a arranement method?
    print ('\t setting face attribute with shape description ...')    
    for idx,face in enumerate(arrange.decomposition.faces):
        arrange.decomposition.faces[idx].set_shape_descriptor(arrange)
```

```python
# face_low_occ_percent = .05 # self below "low_occ_percent"
# face_high_occ_percent = .1 # no nodes more than "high_occ_percent"
# face_consider_categories = [True, False][0]
# face_similar_thr = 0.4

	######################################## face growing the arrangement
    print ('\t face growing the ...') 
    arrange = mapali.prune_arrangement_with_face_growing (arrange, label_image,
                                            low_occ_percent=face_low_occ_percent,
                                            high_occ_percent=face_high_occ_percent,
                                            consider_categories=face_consider_categories,
                                            similar_thr=face_similar_thr)

    ######################################## updating faces label
    # due to the changes of the arrangement and faces
    print ('\t update place categories to faces assignment ...')
    mapali.assign_label_to_all_faces(arrangements[key], label_images[key])    
```

```python

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

```



```python
################################################################################
###################################################### CPD: coherent point drift
################################################################################
'''
What to use? nodes from arrangement.prime or point samples from occupancy?
if using occupancy, I can also try daniels work with distance image from skiz computation
'''

# src: layout
# dst: tango
point_set = ['arrangement', 'point_cloud'][0]

if point_set == 'arrangement':
    arr_src = arrangements[keys[0]]
    arr_dst = arrangements[keys[1]]
    src = [arr_src.graph.node[key]['obj'].point for key in arr_src.graph.node.keys()]
    src = np.array([ [float(p.x),float(p.y)] for p in src ])
    dst = [arr_dst.graph.node[key]['obj'].point for key in arr_dst.graph.node.keys()]
    dst = np.array([ [float(p.x),float(p.y)] for p in dst ])

elif point_set == 'point_cloud':
    # nonzero returns ([y,..],[x,...])
    # flipped to have (x,y)    
    img1 = np.flipud( cv2.imread( mapali.data_sets[keys[0]] , cv2.IMREAD_GRAYSCALE) ) 
    img2 = np.flipud( cv2.imread( mapali.data_sets[keys[1]] , cv2.IMREAD_GRAYSCALE) )
    skip = 20
    src = np.fliplr( np.transpose(np.nonzero(img1<10)) )[0:-1:skip]
    dst = np.fliplr( np.transpose(np.nonzero(img2<10)) )[0:-1:skip]
print (src.shape, dst.shape)

# np.save('src_'+point_set, src)
# np.save('dst_'+point_set, dst)
# np.save('rot', rot_mat(tf.rotation))
# np.save('tra', np.atleast_2d(tf.translation))
# np.save('sca', tf.scale[0])

#### plotting input point sets
mapali.plot_point_sets (src, dst) # src:blue, dst:red

### transforming before registeration
# src_warp = tf._apply_mat(src, tf.params)
rot_mat = lambda t: np.array([[np.cos(t), -np.sin(t)],[np.sin(t), np.cos(t)]])


### registration
fig = plt.figure()
fig.add_axes([0, 0, 1, 1])
callback = partial(mapali.visualize, ax=fig.axes[0])
reg = PCD.RigidRegistration( dst,src,
                             R=rot_mat(tf.rotation),
                             t=np.atleast_2d(tf.translation),
                             s=tf.scale[0],
                             sigma2=None, maxIterations=100, tolerance=0.001)
# reg = AffineRegistration(dst, src)
# reg = DeformableRegistration(dst, src)
Y_transformed, s, R, t = reg.register(callback)
plt.show()
print (reg.err)
```

```python
########################################
################# optimizing with points
########################################


# # src: layout
# # dst: tango
point_set = ['arrangement', 'point_cloud'][0]

if point_set == 'arrangement':
    arr_src = arrangements[keys[0]]
    arr_dst = arrangements[keys[1]]
    src = [arr_src.graph.node[key]['obj'].point for key in arr_src.graph.node.keys()]
    src_point_set = np.array([ [float(p.x),float(p.y)] for p in src ])
    dst = [arr_dst.graph.node[key]['obj'].point for key in arr_dst.graph.node.keys()]
    dst_point_set = np.array([ [float(p.x),float(p.y)] for p in dst ])

elif point_set == 'point_cloud':
    # nonzero returns ([y,..],[x,...]) -> flipped lr to have (x,y)    
    img1 = np.flipud( cv2.imread( mapali.data_sets[keys[0]] , cv2.IMREAD_GRAYSCALE) ) 
    img2 = np.flipud( cv2.imread( mapali.data_sets[keys[1]] , cv2.IMREAD_GRAYSCALE) )
    skip = 20
    src_point_set = np.fliplr( np.transpose(np.nonzero(img1<10)) )[0:-1:skip]
    dst_point_set = np.fliplr( np.transpose(np.nonzero(img2<10)) )[0:-1:skip]
print (src_point_set.shape, dst_point_set.shape)


# src_point_set = np.load('examples/aligne_optimize/CPD_E5_05_tango/src_point_cloud.npy')
# dst_point_set = np.load('examples/aligne_optimize/CPD_E5_05_tango/dst_point_cloud.npy')

sigma = 1.5 # from neighborhood size n=5 -> sigma = 0.3(n/2 - 1) + 0.8
X0 = (tf.translation[0], tf.translation[1], tf.scale[0], tf.rotation)
X_bounds = ((None,None),(None,None),(None,None),(None,None)) # No bounds
X_bounds = ((X0[0]-100,X0[0]+100),(X0[1]-100,X0[1]+100),
            (X0[2]-.1, X0[2]+.1), (X0[3]-.08,X0[3]+.08))
methods = [ 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
         'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']#[1,3,5,7,8,]
# [4,9,10]: need jac
# [0,2,6]: did not converge

result = scipy.optimize.minimize(mapali.objectivefun_pointset, X0,
                                 args=(src_point_set, dst_point_set, sigma),
                                  method = methods[0],
                                 # tol=1e-5,
                                 options={'maxiter':100, 'disp':True} )


if result['success']:
    tx,ty,s,t = result['x']
    print (tx,ty,s,t)

    tf_opt = skimage.transform.AffineTransform(scale=(s,s), rotation=t,
                                             translation=(tx,ty))

    fig, axes = plt.subplots(1,2, figsize=(20,12))
    match_score_ini = mapali.pointset_match_score (src_point_set, dst_point_set,
                                                   sigma, tf)
    title_ini = 'initial (match score:{:.2f})'.format(match_score_ini)
    axes[0] = mapali.plot_transformed_images( images[keys[0]], images[keys[1]],
                                              tformM=tf.params,
                                              axes=axes[0],
                                              title=title_ini)
    match_score_opt = mapali.pointset_match_score (src_point_set, dst_point_set,
                                                   sigma, tf_opt)
    title_opt = 'initial (match score:{:.2f})'.format(match_score_opt)
    axes[1] = mapali.plot_transformed_images( images[keys[0]], images[keys[1]],
                                              tformM=tf_opt.params,
                                              axes=axes[1],
                                              title=title_opt)
    plt.tight_layout()
    plt.show()

    # plt.plot(dst[:,0],dst[:,1], 'b.')
    # src_warp_opt = tf_opt._apply_mat(src,tf_opt.params)
    # plt.plot(src_warp[:,0],src_warp[:,1], 'r.')
    # src_warp_ini = tf._apply_mat(src,tf.params)
    # plt.plot(src_warp_ini[:,0],src_warp_ini[:,1], 'g.')
    # plt.show()

```


```python
##### find 

### version 1 - first transform arrangement and then compute match-score
tf = good_cluster[0]
tic = time.time()
arrange_src = copy.deepcopy(arrangements[keys[0]])
arrange_dst = copy.deepcopy(arrangements[keys[1]])
arrange_src.transform_sequence( operTypes='SRT',
                                operVals=( tf.scale, tf.rotation, tf.translation ),
                                operRefs=( (0,0),    (0,0),       (0,0)  ) )
s = mapali.arrangement_match_score(arrange_src, arrange_dst)
print ( 'version 1: score={:.4f}, time={:.4f}'.format(s, time.time()-tic) )

### version 2 - just transform faces (and superfaces) and then compute match-score
tf = good_cluster[0]
tic = time.time()
arrange_src = copy.deepcopy(arrangements[keys[0]])
arrange_dst = copy.deepcopy(arrangements[keys[1]])
s = mapali.arrangement_match_score_fast(arrange_src, arrange_dst, tf)
print ( 'version 2: score={:.4f}, time={:.4f}'.format(s, time.time()-tic) )

```


```python
### face to face association and match score + plotting
# face to face association
f2f_association = mapali.find_face2face_association(arrange_src,
                                                    arrange_dst,
                                                    distance='area')

# face to face match score (associated faces)
# note: since each face is dealt with once, this is [almost] OK
# otherwise, caching the "pixels_in_face" would improve speed. 
f2f_match_score = {(src_idx,f2f_association[src_idx]): None
                   for src_idx in f2f_association.keys()}
for (src_idx,dst_idx) in f2f_match_score.keys():
    score = mapali.face_match_score(arrange_src.decomposition.faces[src_idx],
                                    arrange_dst.decomposition.faces[dst_idx])
    f2f_match_score[(src_idx,dst_idx)] = score

mapali.plot_face2face_association_match_score(arrange_src, arrange_dst,
                                              f2f_association, f2f_match_score)

```


```python
#############################3### computing the MSE error for each cluster
src_dis_img = dis_images[keys[0]]
dst_dis_img = dis_images[keys[1]]

cluster_error = {}
for lbl in unique_labels:
    if lbl != -1:
        class_member_idx = np.nonzero(labels == lbl)[0]
        class_member = [ tforms[idx]
                         for idx in class_member_idx ]

        err = [ mapali.mse_norm(src_dis_img, dst_dis_img, tf) 
                for tf in class_member ]
        
        cluster_error[lbl] = err #np.mean(err)

for lbl in unique_labels:
    if lbl != -1:
        print (lbl, cluster_error[lbl])
```

```python
####### changes of basis to match opencv
# not solved yet
A = np.array([[1,0,0],[0,-1,0],[0,0,1]])
M = np.dot(A,tf.params)
Mp = np.dot(A,M)

mirr_tf = skimage.transform.AffineTransform(M)

srcimg = cv2.imread( mapali.data_sets[keys[0]], cv2.IMREAD_GRAYSCALE)
dstimg = cv2.imread( mapali.data_sets[keys[1]], cv2.IMREAD_GRAYSCALE)

aff2d = matplotlib.transforms.Affine2D( mirr_tf.params )
aff2d = matplotlib.transforms.Affine2D( tf.params )
fig, axes = plt.subplots(1,1, figsize=(20,12))
im_dst = axes.imshow(dstimg, origin='upper', cmap='gray', alpha=.5, clip_on=True)
im_src = axes.imshow(srcimg, origin='upper', cmap='gray', alpha=.5, clip_on=True)
# im_src.set_transform( matplotlib.transforms.Affine2D( A ) + axes.transData )
# im_src.set_transform( matplotlib.transforms.Affine2D( tf.params ) + axes.transData )
# im_src.set_transform( matplotlib.transforms.Affine2D( A ) + axes.transData )
im_src.set_transform( aff2d + axes.transData )

plt.tight_layout()
plt.show()
```


```python
######################################## face growing the arrangement
# for every neighboruing faces, merge if:
#     1. faces are neighboring with same place categies
#     2. the mutual edges are in the list `edges_to_purge`
#     3. the emerging face must have the same shape as initial faces

face_grow_similar_thr = 0.4

done_growing = False
while not done_growing :

    # unless a new pair of faces are merged, we assume we are done merging
    done_growing = True 
    
    # set node and edge occupancy values
    mapali.set_node_occupancy(arrange, image,
                              occupancy_thr=prun_image_occupancy_thr,
                              neighborhood=prun_node_neighborhood,
                              attribute_key='occupancy')
    
    mapali.set_edge_occupancy(arrangement, image,
                              occupancy_thr=prun_image_occupancy_thr,
                              neighborhood=prun_edge_neighborhood,
                              attribute_key='occupancy')
    
    # be almost generous with this, not too much, this is like a green light
    # the faces won't merge unless they have same category and same shape 
    edges_to_purge = get_edges_to_purge (arrange,
                                         low_occ_percent=prun_low_occ_percent,
                                         high_occ_percent=prun_high_occ_percent,
                                         consider_categories=prun_consider_categories )


    for f1_idx in range(len(arrange.decomposition.faces)):
        for f2_idx in arrange.decomposition.find_neighbours(f1_idx):
            f1 = arrange.decomposition.faces[f1_idx]
            f2 = arrange.decomposition.faces[f2_idx]
            
            # checking if faces are similar (category label)
            similar_label = mapali.are_same_category(f1,f2,
                                                     label_associations=None,
                                                     thr=face_grow_similar_thr)
            
            # checking if faces are similar (shape)
            f1.set_shape_descriptor(arrange, remove_redundant_lines=True)
            f2.set_shape_descriptor(arrange, remove_redundant_lines=True)
            similar_shape = True if len(utls.match_face_shape(f1,f2))>0 else False
            
            # cheking if it's ok to prun mutual halfedges
            mut_he = arrange.decomposition.find_mutual_halfEdges(f1_idx,f2_idx)
            ok_to_prun = all([he in edges_to_purge for he in mut_he])
            
            if similar_label and similar_shape and ok_to_prun:
                
                new_face = utls.merge_faces_on_fly(arrange, f1_idx, f2_idx)
                if new_face is not None:
                    
                    # checking if the new face has similar shape to originals
                    new_face.set_shape_descriptor(arrange, remove_redundant_lines=True)
                    if len(utls.match_face_shape(new_face,f2))>0:
                        done_growing = False
                        arrange.remove_edges(mut_he, loose_degree=2)
```


```python
# constructing the desired transform [E5_01]
t = skimage.transform.AffineTransform( scale=(1.2,1.2), 
                                       rotation=np.pi/2+0.04,
                                       translation=(1526,15) )
```

```python
# finding targets for [E5_01] (i.e. transforms closet to desired)
target = []
for idx,tf in enumerate(tforms):
    dt = np.sqrt( np.sum( (tf.translation - np.array([1526,15]))**2 ) )
    ds = np.abs(tf.scale - 1.2)
    dr = np.abs(tf.rotation - (np.pi/2+0.04))
    if (ds <0.1) and (dr <0.1) and (dt <200):
        target.append(idx)
print (len(target))
```

```python
# checking the variance of parameters in each cluster
for lbl in unique_labels:
    if lbl != -1:
        class_member_idx = np.nonzero(labels == lbl)[0]
        class_member = [ tforms[idx]
                         for idx in class_member_idx ]
        t_var = np.var([ tf.translation
                           for tf in class_member ], axis=0)
        s_var = np.var([ tf.scale
                           for tf in class_member ])
        r_var = np.var([ tf.rotation
                           for tf in class_member ])        
        msg = 'cluster {:d}: t:({:.2f},{:.2f}) - s:{:.2f} - r:{:.2f}'
        print (msg.format(lbl,t_var[0],t_var[1],s_var,r_var ))
# translation has very high variance, but it's actually ok!
# they are not that far, and the high variance seems wierd
class_member_idx = np.nonzero(labels == 35)[0]
class_member = [ tforms[idx]
                 for idx in class_member_idx ]
for tf in class_member: print (tf.translation)
```

```python
# plotting the histogram of parameters' distributions 
fig, axes = plt.subplots(1,1, figsize=(20,12))
axes.hist(parameters[:,0], facecolor='b', bins=100, alpha=0.7, label='tx')
axes.hist(parameters[:,1], facecolor='r', bins=100, alpha=0.7, label='ty')
# axes.hist(parameters[:,2], facecolor='g', bins=100, alpha=0.7, label='rotate')
# axes.hist(parameters[:,3], facecolor='m', bins=100, alpha=0.7, label='scale')
axes.legend(loc=1, ncol=1)
axes.set_title('histogram of alignment parameters')
plt.tight_layout()
plt.show()
```

```python
# plotting all transformations and targets in 1)"transformed unit-vector" space and 2)features space (tx-ty / r-s) 
fig, axes = plt.subplots(1,1, figsize=(20,12))
U = np.array([1,1,1])
# "transformed unit-vector" space
xy = np.stack([ np.dot(tforms[idx].params, U)[:2]
                for idx in range(len(tforms)) ], axis=0)
axes.plot(xy[:, 0], xy[:, 1], ',',
          markerfacecolor='k', markeredgecolor='k')
xy = np.stack([ np.dot(tforms[idx].params, U)[:2]
                for idx in target ], axis=0)
axes.plot(xy[:, 0], xy[:, 1], '.',
          markerfacecolor='r', markeredgecolor='r')
# features space (tx-ty)
xy = np.stack([ tforms[idx].translation
                for idx in range(len(tforms)) ], axis=0)
axes.plot( xy[:, 0], xy[:, 1], ',',
           markerfacecolor='k', markeredgecolor='k')
xy = np.stack([ tforms[idx].translation
                for idx in target ], axis=0)
axes.plot( xy[:, 0], xy[:, 1], '.',
           markerfacecolor='r', markeredgecolor='r')
# features space (r-s)
xy = np.stack([ (tforms[idx].scale, tforms[idx].rotation)
                for idx in range(len(tforms)) ], axis=0)
axes.plot( xy[:, 0], xy[:, 1], ',',
           markerfacecolor='k', markeredgecolor='k')
xy = np.stack([ (tforms[idx].scale, tforms[idx].rotation)
                for idx in target ], axis=0)
axes.plot( xy[:, 0], xy[:, 1], '.',
           markerfacecolor='r', markeredgecolor='r')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

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
