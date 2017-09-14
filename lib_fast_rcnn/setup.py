# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from distutils.core import setup
from setuptools import find_packages, Extension

setup(
    name='lib_fast_rcnn',
    packages=find_packages(),
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.so'],
    },
)
