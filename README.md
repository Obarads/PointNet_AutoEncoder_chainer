# chainer-pointnet-autoencoder
## Introduce
AutoEncoder using PointNet and chainer. PointNet-AutoEnocoder(TensorFlow) which PointNet author created is [here](https://github.com/charlesq34/pointnet-autoencoder).

## Installation
Please install Chainer (and cupy if you want to use GPU) beforehand.  
Furthermore, Operation check is described on comments.
```
# chainer version 5.0.0
pip install chainer
# cupy-cuda92 version 5.0.0
pip install cupy-cuda92
```
Also, some extension library is used in some of the code,
```
# Chainer Chemistry version 0.4.0
git clone https://github.com/pfnet-research/chainer-chemistry.git
pip install -e chainer-chemistry
# ChainerEX version 0.0.1
git clone https://github.com/corochann/chainerex.git
pip install -e chainerex
```

## Download Data
ShapeNetPart dataset is available [HERE](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip). Simply download the zip file and move the `shapenetcore_partanno_segmentation_benchmark_v0` folder to `data`.

Besides this, you can use command to download ShapeNetPart dataset.
```
python chainer_dataset.py -d true
```

## Train
You can simply execute train code with GPU.
```
python train.py -g 0
```

## Check using trained model and viewer
You can view a ground truth data and then view an output data. In order to view next data, Please press q.
```
python test.py
```

## References
1. Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR 2017.
1. "charlesq34/pointnet-autoencoder: Autoencoder for Point Clouds". Github. https://github.com/charlesq34/pointnet-autoencoder, (accessed 2018-12-5).
1. "corochann_chainer-pointnet_ Chainer implementation of PointNet, PointNet++, KD-Network and 3DContextNework". Github. https://github.com/corochann/chainer-pointnet, (accessed 2018-11-13).
