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
import models.oneclasspn as ocpn
from models import dataset as ds


def main():

    # model setting

    seed = 0
    num_point = 200
    out_dir = 0 #no use
    debug = False
    batch_size = 32

    out_dim = 3
    in_dim = 3
    middle_dim = 64
    dropout_ratio = 0
    use_bn = True
    trans = True
    trans_lam1 = 0.001
    trans_lam2 = 0.001
    compute_accuracy = True
    residual = False
    print('Train OneClassPN model... trans={} use_bn={} dropout={}'
        .format(trans, use_bn, dropout_ratio))
    model = ocpn.OneClassPN(out_dim=out_dim, in_dim=in_dim, middle_dim=middle_dim, dropout_ratio=dropout_ratio, use_bn=use_bn,
                            trans=trans, trans_lam1=trans_lam1, trans_lam2=trans_lam2, compute_accuracy=compute_accuracy, residual=residual)

    # Dataset preparation
    train = ds.get_train_dataset(num_point=num_point)
    val = ds.get_test_dataset(num_point=num_point)

    train_iter = iterators.SerialIterator(train, batch_size)
    val_iter = iterators.SerialIterator(
        val, batch_size, repeat=False, shuffle=False)
    device = 0
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(
        train_iter, optimizer, device=device, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


if __name__ == '__main__':
    main()
