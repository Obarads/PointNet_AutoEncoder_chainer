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
    x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]]).astype('f') 
    t = np.array([3, 0]).astype('i') 
    y = functions.softmax_cross_entropy(x, t)
    print(y)
    h = functions.matmul(t, x[:, :, :, 0])

if __name__ == '__main__':
    main()
