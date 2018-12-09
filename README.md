# chainer-pointnet-autoencoder
## Introduce
AutoEncoder using PointNet and chainer. PointNet-AutoEnocoder(TensorFlow) which PointNet author created is [here](https://github.com/charlesq34/pointnet-autoencoder).

## Installation
Please install Chainer (and cupy if you want to use GPU) beforehand.  
Operation check is as follows:
- chainer 5.0.0
- cupy-cuda92 5.0.0  

Also, some extension library is used in some of the code,
```
# Chainer Chemistry
git clone https://github.com/pfnet-research/chainer-chemistry.git
pip install -e chainer-chemistry
# ChainerEX
git clone https://github.com/corochann/chainerex.git
pip install -e chainerex
```
## References
1. Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR 2017.
1. "charlesq34/pointnet-autoencoder: Autoencoder for Point Clouds". Github. https://github.com/charlesq34/pointnet-autoencoder, (accessed 2018-12-5).
1. "corochann_chainer-pointnet_ Chainer implementation of PointNet, PointNet++, KD-Network and 3DContextNework". Github. https://github.com/corochann/chainer-pointnet, (accessed 2018-11-13).
