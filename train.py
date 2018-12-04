import chainer
from chainer import serializers
from chainer import functions
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.dataset import to_device, concat_examples
from chainer.datasets import TransformDataset
from chainer.training import extensions as E
from chainer.dataset.convert import concat_examples as converter

import numpy as np
import os

# self made
import models.oneclasspn as ocpn
import dataset as ds


def main():

    # Network
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
    seed = 888
    num_point = 200
    batch_size = 32
    train = ds.get_train_dataset(num_point=num_point)
    val = ds.get_test_dataset(num_point=num_point)
    train_iter = iterators.SerialIterator(train, batch_size)
    val_iter = iterators.SerialIterator(
        val, batch_size, repeat=False, shuffle=False)


    # gpu setting
    device = 0
    print('using gpu {}'.format(device))
    chainer.backends.cuda.get_device_from_id(device).use()
    model.to_gpu()


    # Optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    # traning
    epoch = 250
    out_dir = 'result'
    updater = training.StandardUpdater(
        train_iter, optimizer, device=device, converter=converter)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_dir)

    from chainerex.training.extensions import schedule_optimizer_value
    from chainer.training.extensions import observe_value
    # trainer.extend(observe_lr)
    observation_key = 'lr'
    trainer.extend(observe_value(
        observation_key,
        lambda trainer: trainer.updater.get_optimizer('main').alpha))
    trainer.extend(schedule_optimizer_value(
        [10, 20, 100, 150, 200, 230],
        [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]))

    trainer.extend(E.Evaluator(
        val_iter, model, converter=converter, device=device))
    trainer.extend(E.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.PrintReport(
        ['epoch', 'main/loss', 'main/cls_loss', 'main/trans_loss1',
         'main/trans_loss2', 'main/accuracy', 'validation/main/loss',
         # 'validation/main/cls_loss',
         # 'validation/main/trans_loss1', 'validation/main/trans_loss2',
         'validation/main/accuracy', 'lr', 'elapsed_time']))
    trainer.extend(E.ProgressBar(update_interval=10))

    resume = ''
    if resume:
        serializers.load_npz(resume, trainer)
    trainer.run()

    # --- save classifier ---
    # protocol = args.protocol
    # classifier.save_pickle(
    #     os.path.join(out_dir, args.model_filename), protocol=protocol)
    model_filename = 'model.npz'
    serializers.save_npz(
        os.path.join(out_dir, model_filename), model)


if __name__ == '__main__':
    main()
