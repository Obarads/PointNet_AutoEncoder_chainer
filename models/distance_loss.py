# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import functions
from chainer import Variable

"""
reference charlesq34/pointnet-autoencoder
website:https://github.com/charlesq34/pointnet-autoencoder/blob/master/tf_ops/nn_distance/tf_nndistance_cpu.py
(access:2018/12/5 (yyyy/mm/dd))
"""

def chamfer_distance(pc1, pc2):
    '''
    Input:
        pc1: float chainer in shape (B,N,C) the first point cloud
        pc2: float chainer in shape (B,M,C) the second point cloud
    Output:
        dist1: float chainer in shape (B,N) distance from first to second 
        idx1: int32 chainer in shape (B,N) nearest neighbor from first to second
        dist2: float chainer in shape (B,M) distance from second to first 
        idx2: int32 chainer in shape (B,M) nearest neighbor from second to first
    '''

    dist1,idx1,dist2,idx2=0,0,0,0

    N = pc1.shape[2]
    M = pc2.shape[2]

    dist = Variable(np.zeros((N,M)))
    for i in range(N):
        for j in range(M):
            dist[i,j] = functions.sum((pc1[0,:,i,0] - pc2[0,:,j,0]) ** 2)
    dist1 = functions.min(dist,axis=1)
    dist2 = functions.min(dist,axis=1)

    """
    #行列のn列目に1次元挿入
    #各次元を指定配列分だけ繰り返す、上の場合、２次元の部分をM回繰り返している。
    pc1_expand_tile = functions.tile(functions.expand_dims(pc1,3),(1,1,1,M,1))
    pc2_expand_tile = functions.tile(functions.expand_dims(pc2,2),(1,1,N,1,1))
    #pc1_expand_tile = functions.tile(pc1,(1,1,M,1))
    #pc2_expand_tile = functions.tile(pc2,(1,N,1,1))
    #pc1_expand_tile shape = pc2_expand_tile shape

    #pc_diff is difference between pc1 and pc2 in coordinate system.
    #print(pc1_expand_tile.shape)
    #print(pc2_expand_tile.shape)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    pc_dist = functions.sum(pc_diff**2, axis=1)

    dist1 = functions.min(pc_dist,axis=1)
    #idx1 = functions.argmin(pc_dist, axis=1)
    dist2 = functions.min(pc_dist,axis=2)
    #idx2 = functions.argmin(pc_dist, axis=2)
    """

    return dist1, idx1, dist2, idx2


def verify_chamfer_distance_cup():
    np.random.seed(0)
    pc1arr = np.random.random((1,3,5,1))
    pc2arr = np.random.random((1,3,6,1))
    pc1 = Variable(pc1arr)
    pc2 = Variable(pc2arr)

    """
    print("pc1arr:{}".format(pc1arr))
    print("pc2arr:{}".format(pc2arr))
    print("pc1:{}".format(pc1))
    print("pc2:{}".format(pc2))
    """

    dist1, idx1, dist2, idx2 = chamfer_distance(pc1, pc2)
    print("dist1:{}".format(dist1))
    print("dist2:{}".format(dist2))
    
    #print("pc1arr:{}".format(pc1arr))
    #print("pc2arr:{}".format(pc2arr))

    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            dist[i,j] = np.sum((pc1arr[0,:,i,0] - pc2arr[0,:,j,0]) ** 2)
            
    print("dist")
    print(dist)

if __name__ == '__main__':
    verify_chamfer_distance_cup()
