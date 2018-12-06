import chainer
from chainer import functions
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.dataset import to_device, concat_examples
from chainer.datasets import TransformDataset
from chainer.training import extensions

import numpy as np

# self made

def main():
    data = [
        [
            [0,0,0],
            [1,1,1],
            [2,2,2]
        ],[
            [0,0,0],
            [1,1,1],
            [2,2,2]
        ]
    ]
    l = len(data[0][0])
    print(l)
    print(data[:, :l, :].astype(np.float32))

if __name__ == '__main__':
    main()
