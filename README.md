# Map-Alignment-2D


## Installation and dependencies



## Usage


### Interpretations:

* For 3D-mesh maps:
  ```
  ogm = convert_to_2D(ply)
  yaml = trait_detect(ogm)
  dist = distance_transform(ogm)
  # plcat = place_categories(ogm)
  ```

* For layout maps:
  ```
  ogm = convert_to_bitmap(svg) # if use inkscape, beware the origin is set at "top"
  yaml = parse_svg_to_yaml(svg)
  dist = distance_transform(ogm)
  # plcat = place_categories(ogm)
  ```

### Alignment:


