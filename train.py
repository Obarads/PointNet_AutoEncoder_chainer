import chainer
from chainer import functions
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.dataset import to_device, concat_examples
from chainer.datasets import TransformDataset
from chainer.training import extensions

import numpy as np

#self made
import models.oneclasspn as ocpn


def main():

    # model setting
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
    model = ocpn.OneClassPN(out_dim=out_dim, in_dim=in_dim, middle_dim=middle_dim, dropout_ratio=dropout_ratio, use_bn=use_bn,
                    trans=trans, trans_lam1=trans_lam1, trans_lam2=trans_lam2, compute_accuracy=compute_accuracy, residual=residual)

    # gpu setting
    train_iter = iterators.SerialIterator(train, args.batchsize)
    val_iter = iterators.SerialIterator(
        val, args.batchsize, repeat=False, shuffle=False)
    device = 1
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)


if __name__ == '__main__':
    main()
