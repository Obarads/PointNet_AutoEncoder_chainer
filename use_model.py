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
import chainer_dataset as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        description='AutoEncoder ShapeNet')
    parser.add_argument('--dropout_ratio', type=float, default=0)
    parser.add_argument('--trans', type=strtobool, default='true')
    parser.add_argument('--use_bn', type=strtobool, default='true')
    parser.add_argument('--residual', type=strtobool, default='false')
    parser.add_argument('--out_dim', type=int, default=3)
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--middle_dim', type=int, default=64)
    parser.add_argument('--load_file', '-lf', type=str, default='result/model.npz')
    parser.add_argument('--class_choice', type=str, default='Chair')
    args = parser.parse_args()

    dropout_ratio = args.dropout_ratio
    trans = args.trans
    use_bn = args.use_bn
    residual = args.residual
    out_dim = args.out_dim
    in_dim = args.in_dim
    middle_dim = args.middle_dim
    class_choice = args.class_choice
    load_file = args.load_file

    trans_lam1 = 0.001
    trans_lam2 = 0.001

    print('Load PointNet-AutoEncoder model... load_file={}'.format(load_file))
    model = ae.PointNetAE(out_dim=out_dim, in_dim=in_dim, middle_dim=middle_dim, dropout_ratio=dropout_ratio, use_bn=use_bn,
                          trans=trans, trans_lam1=trans_lam1, trans_lam2=trans_lam2, residual=residual)
    serializers.load_npz(load_file, model)

    d = pd.ChainerDataset(root=os.path.join(
        BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0'), split="test", class_choice=[class_choice])
    x,_ = d.get_example(0)
    x = chainer.Variable(np.array([x]))
    print(x.shape)
    h = model.test_calc(x)

    #note: update "for" to transpose
    point_data = np.transpose(h[0].astype(np.float32), (1, 0))[:, :, None]

    import utils.show3d_balls as show3d_balls
    show3d_balls.showpoints(point_data[0], ballradius=8)

if __name__ == '__main__':
    main()