Map-Alignment-2D
================


Dependencies and Download
-------------------------

- Download, installing dependencies, and install package

```shell
# Download
git clone https://github.com/saeedghsh/arrangement/
cd arrangement

# Install dependencies
pip install -r requirements.txt
pip3 install -r requirements.txt

# Install the package
python setup.py install
python3 setup.py install
```

Most dependencies are listed in ```requirements.txt```.
But there are three more, namely [opencv](http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html), [Polygon](https://www.j-raedler.de/projects/polygon/) and [arrangement](https://github.com/saeedghsh/arrangement/), which should be installed separately.



Usage Example
-------------
- Demo
```shell

```

![animation](https://github.com/saeedghsh/arrangement/blob/master/docs/animation.gif)


Setting Of The Parameters
-------------------------



License
-------
Distributed with a GNU GENERAL PUBLIC LICENSE; see LICENSE.
```
Copyright (C) Saeed Gholami Shahbandi <saeed.gh.sh@gmail.com>
```
This package has been developed to be employed as the underlying spatial representation for robot maps in the following publications:
- S. G. Shahbandi, M. Magnusson, "2D Map Alignment With Region Decomposition", submitted to Autonomous Robots, 2017.


Laundry List
------------
- [ ] move new methods from ```demo.py``` to ```mapali```.
- [ ] remove unused methods from ```mapali``` and ```plotting```.
- [ ] documentation.
- [ ] full test suite.
- [ ] profile for speed-up.
- [ ] python3 compatible.



