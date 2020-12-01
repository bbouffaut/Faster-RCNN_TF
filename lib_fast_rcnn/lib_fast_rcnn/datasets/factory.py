# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

import datasets
import lib_fast_rcnn.datasets.pascal_voc
import lib_fast_rcnn.datasets.imagenet3d
import lib_fast_rcnn.datasets.kitti
import lib_fast_rcnn.datasets.kitti_tracking
import numpy as np


def _selective_search_IJCV_top_k(split, year, top_k, classes):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year, classes)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

class factory():

    def __init__(self,classes):

        print('DEBUG datasets_factory {}'.format(classes))

        self._sets = {}

         # Set up voc_<year>_<split> using selective search "fast" mode
        for year in ['2007', '2012']:
            for split in ['train', 'val', 'trainval', 'test']:
                name = 'voc_{}_{}'.format(year, split)
                self._sets[name] = (lambda split=split, year=year:
                        datasets.pascal_voc(split, year, classes))
            """
        # Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
        # but only returning the first k boxes
        for top_k in np.arange(1000, 11000, 1000):
            for year in ['2007', '2012']:
                for split in ['train', 'val', 'trainval', 'test']:
                    name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
                    self._sets[name] = (lambda split=split, year=year, top_k=top_k:
                            _selective_search_IJCV_top_k(split, year, top_k))
        """

        # Set up voc_<year>_<split> using selective search "fast" mode
        for year in ['2007']:
            for split in ['train', 'val', 'trainval', 'test']:
                name = 'voc_{}_{}'.format(year, split)
                print name
                self._sets[name] = (lambda split=split, year=year:
                        datasets.pascal_voc(split, year, classes))

        # KITTI dataset
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'kitti_{}'.format(split)
            print name
            self._sets[name] = (lambda split=split:
                    datasets.kitti(split))

        # Set up coco_2014_<split>
        for year in ['2014']:
            for split in ['train', 'val', 'minival', 'valminusminival']:
                name = 'coco_{}_{}'.format(year, split)
                self._sets[name] = (lambda split=split, year=year: coco(split, year))

        # Set up coco_2015_<split>
        for year in ['2015']:
            for split in ['test', 'test-dev']:
                name = 'coco_{}_{}'.format(year, split)
                self._sets[name] = (lambda split=split, year=year: coco(split, year))

        # NTHU dataset
        for split in ['71', '370']:
            name = 'nthu_{}'.format(split)
            print name
            self._sets[name] = (lambda split=split:
                    datasets.nthu(split))


    def get_imdb(self,name):
        """Get an imdb (image database) by name."""
        if not self._sets.has_key(name):
            raise KeyError('Unknown dataset: {}'.format(name))
        return self._sets[name]()

    def list_imdbs(self):
        """List all registered imdbs."""
        return self._sets.keys()
