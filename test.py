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
    np.random.seed(0)

    data = np.arange(3)
    data = chainer.Variable(data)
    ite  = chainer.iterators.SerialIterator(data, batch_size=2, 
                                            repeat=True, shuffle=True)

    print(data)

    for i in range(5):
        print(ite.next())

if __name__ == '__main__':
    main()
