import chainer
from chainer import serializers
from chainer import functions
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.dataset import to_device
from chainer.datasets import TransformDataset
from chainer.training import extensions as E
from chainer.dataset.convert import concat_examples
from chainer.datasets.concatenated_dataset import ConcatenatedDataset

import numpy as np
import os
import argparse
from distutils.util import strtobool

# self made
import models.pointnet_ae as ae
import part_dataset as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description='AutoEncoder ShapeNet')
    # parser.add_argument('--conv-layers', '-c', type=int, default=4)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--dropout_ratio', type=float, default=0)
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=250)
    # parser.add_argument('--unit-num', '-u', type=int, default=16)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--protocol', type=int, default=2)
    parser.add_argument('--model_filename', type=str, default='model.npz')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--trans', type=strtobool, default='true')
    parser.add_argument('--use_bn', type=strtobool, default='true')
    parser.add_argument('--normalize', type=strtobool, default='false')
    parser.add_argument('--residual', type=strtobool, default='false')
    parser.add_argument('--out_dim', type=int, default=3)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--middle_dim', type=int, default=64)
    parser.add_argument('--use_val', type=strtobool, default='false')
    args = parser.parse_args()

    batch_size = args.batchsize
    dropout_ratio = args.dropout_ratio
    num_point = args.num_point
    device = args.gpu
    out_dir = args.out
    epoch = args.epoch
    seed = args.seed
    protocol = args.protocol
    model_filename = args.model_filename
    resume = args.resume
    trans = args.trans
    use_bn = args.use_bn
    normalize = args.normalize
    residual = args.residual
    out_dim = args.out_dim
    in_dim = args.in_dim
    middle_dim = args.middle_dim
    use_val = args.use_val

    trans_lam1 = 0.001
    trans_lam2 = 0.001

    try:
        os.makedirs(out_dir, exist_ok=True)
        import chainerex.utils as cl
        fp = os.path.join(out_dir, 'args.json')
        cl.save_json(fp, vars(args))
        print('save args to', fp)
    except ImportError:
        pass

    # Network
    print('Train PointNet-AutoEncoder model... trans={} use_bn={} dropout={}'
          .format(trans, use_bn, dropout_ratio))
    model = ae.PointNetAE(out_dim=out_dim, in_dim=in_dim, middle_dim=middle_dim, dropout_ratio=dropout_ratio, use_bn=use_bn,
                          trans=trans, trans_lam1=trans_lam1, trans_lam2=trans_lam2, residual=residual)

    print("Dataset setting...")
    # Dataset preparation
    train = ConcatenatedDataset(*([pd.ChainerDataset(root=os.path.join(
        BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0'), split="train", class_choice=["Car"])]))
    train_iter = iterators.SerialIterator(train, batch_size)
    if use_val:
        val = ConcatenatedDataset(*(pd.ChainerDataset(root=os.path.join(
            BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0'), split="val", class_choice=["Car"])))
        val_iter = iterators.SerialIterator(
            val, batch_size, repeat=False, shuffle=False)

    print("GPU setting...")
    # gpu setting
    if(device >= 0):
        print('using gpu {}'.format(device))
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()

    # Optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # traning
    converter = concat_examples
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

    if use_val:
        trainer.extend(E.Evaluator(val_iter, model,
                                   converter=converter, device=device))
        trainer.extend(E.PrintReport(
            ['epoch', 'main/loss', 'main/cls_loss', 'main/trans_loss1',
             'main/trans_loss2', 'main/accuracy', 'validation/main/loss',
             'validation/main/cls_loss',
             'validation/main/trans_loss1', 'validation/main/trans_loss2',
             'validation/main/accuracy', 'lr', 'elapsed_time']))
    else:
        trainer.extend(E.PrintReport(
            ['epoch', 'main/loss', 'main/cls_loss', 'main/trans_loss1',
             'main/trans_loss2', 'main/accuracy', 'lr', 'elapsed_time']))
    trainer.extend(E.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(E.LogReport())
    trainer.extend(E.ProgressBar(update_interval=10))

    resume = ''
    if resume:
        serializers.load_npz(resume, trainer)
    print("Traning start.")
    trainer.run()

    # --- save classifier ---
    # protocol = args.protocol
    # classifier.save_pickle(
    #     os.path.join(out_dir, args.model_filename), protocol=protocol)
    serializers.save_npz(
        os.path.join(out_dir, model_filename), model)


if __name__ == '__main__':
    main()
