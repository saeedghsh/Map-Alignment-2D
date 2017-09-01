Map-Alignment-2D
================
This package has been developed for 2D Map Alignment With Region Decomposition.
Details of the method are presented in the following publication.
```
S. G. Shahbandi, M. Magnusson, "2D Map Alignment With Region Decomposition", submitted to Autonomous Robots, 2017.
```

Dependencies and Download
-------------------------
Download, installing dependencies, and install package

```shell
# Download
git clone https://github.com/saeedghsh/Map-Alignment-2D.git
cd Map-Alignment-2D

# Install dependencies
pip install -r requirements.txt
<!-- pip3 install -r requirements.txt -->

# Install the package [optional]
python setup.py install
<!-- python3 setup.py install -->
```

Most dependencies are listed in ```requirements.txt```.
But there are three more, namely [opencv](http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html), [Polygon](https://www.j-raedler.de/projects/polygon/) and [arrangement](https://github.com/saeedghsh/arrangement/), which should be installed separately.


Usage Example
-------------
For simplicity and convenience, we assume both maps are provided as bitmap (occupancy grid maps).
For more examples, see [Halmstad Map Collection](https://github.com/saeedghsh/Halmstad-Robot-Maps).
- Demo
```shell
python demo.py --img_src 'tests/maps/map_src.png' --img_dst 'tests/maps/map_dst.png' -multiprocessing -visualize
```
![example](https://github.com/saeedghsh/Map-Alignment-2D/blob/master/docs/maps_map_src__maps_map_dst.png)


<!-- Parameters Setting -->
<!-- ------------------ -->
<!-- - lnl_config -->
<!-- {'binary_threshold_1': 200, # with numpy - for SKIZ and distance -->
<!-- 'binary_threshold_2': [100, 255], # with cv2 - for trait detection -->
<!-- 'traits_from_file': False, # else provide yaml file name -->
<!-- 'trait_detection_source': 'binary_inverted', -->
<!-- 'edge_detection_config': [50, 150, 3], # thr1, thr2, apt_size -->
<!-- 'peak_detect_sinogram_config': [15, 15, 0.15], # [refWin, minDist, minVal] -->
<!-- 'orthogonal_orientations': True} # for dominant orientation detection -->

<!-- - arr_config -->
<!-- {'multi_processing':4, 'end_point':False, 'timing':False, -->
<!-- 'prune_dis_neighborhood': 2, -->
<!-- 'prune_dis_threshold': .075, # home:0.15 - office:0.075 -->
<!-- 'occupancy_threshold': 200} # cell below this is considered occupied -->

<!-- - hyp_config -->
<!-- {'scale_mismatch_ratio_threshold': .3, # .5, -->
<!-- 'scale_bounds': [.5, 2], #[.1, 10] -->
<!-- 'face_occupancy_threshold': .5} -->
    
<!-- - sel_config -->
<!-- {'multiprocessing': multiprocessing, -->
<!-- 'too_many_tforms': 3000, -->
<!-- 'dbscan_eps': 0.051, -->
<!-- 'dbscan_min_samples': 2} -->


License
-------
Distributed with a GNU GENERAL PUBLIC LICENSE; see LICENSE.
```
Copyright (C) Saeed Gholami Shahbandi <saeed.gh.sh@gmail.com>
```

Laundry List
------------
<!-- - [ ] try out 3points distance for tforms and provide it precomputed to clustering. -->
<!-- - [ ] move new methods from ```demo.py``` to ```mapali```. -->
- [ ] dump unused methods from ```mapali``` and ```plotting```.
- [ ] api documentation.
- [ ] full test suite.
- [ ] profile for speed-up.
- [ ] python3 compatible.



