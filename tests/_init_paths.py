# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib_fast_rcnn','lib_fast_rcnn')
add_path(lib_path)

lib_path = osp.join(this_dir, '..','faster_rcnn_tf')
add_path(lib_path)

lib_path = osp.join(this_dir, '..','faster_rcnn_tf','handlers')
add_path(lib_path)
