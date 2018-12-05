import os

import chainer
from chainer.datasets.concatenated_dataset import ConcatenatedDataset

import models.provider as provider
import numpy as np

class InputDataset(chainer.dataset.DatasetMixin):
    """This if partial dataset"""

    def __init__(self, h5_filepath, num_point=1024, augment=False):
        print('loading ', h5_filepath)
        #data, label = provider.loadDataFile(h5_filepath) #provider is selfmade of pointnet
        # Please create label and Datase that build same provider format.
        data, label = provider.loadDataFile(h5_filepath)
        #print(data)
        """structure
        [
            [
                [x y z],
                [x y z],
                ...
                [x y z]
            ],
            ...
            [
                [x y z],
                [x y z],
                ...
                [x y z]
            ]
        ]
        """
        #print(label)
        """
        [
            [label],
            ...
            [label]
        ]
        """
        assert len(data) == len(label)
        # data: (2048, 2048, 3) - (batchsize, point, xyz)
        # Reduce num point here.
        self.data = data[:, :num_point, :].astype(np.float32)
        # (2048,) - (batchsize,)
        self.label = np.squeeze(label).astype(np.int32)
        self.augment = augment
        self.num_point = num_point
        self.length = len(data)
        print('length ', self.length)
            
    def __len__(self):
        """return length of this dataset"""
        return self.length

    def get_example(self, i):
        """Return i-th data"""
        if self.augment:
            rotated_data = provider.rotate_point_cloud(
                self.data[i:i + 1, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            point_data = jittered_data[0]
        else:
            point_data = self.data[i]
        # pint_data (2048, 3): (num_point, k) --> convert to (k, num_point, 1)
        point_data = np.transpose(
            point_data.astype(np.float32), (1, 0))[:, :, None]
        assert point_data.dtype == np.float32
        assert self.label[i].dtype == np.int32
        return point_data, self.label[i]


def get_train_dataset(num_point=1024):
    print('get train num_point ', num_point)
    train_files = "text.csv"
    return ConcatenatedDataset(
        *(InputDataset(filepath, num_point=num_point, augment=True) for filepath in train_files))

def get_test_dataset(num_point=1024):
    print('get test num_point ', num_point)
    test_files = "text.csv"
    return ConcatenatedDataset(
        *(InputDataset(filepath, num_point=num_point, augment=True) for filepath in test_files))
