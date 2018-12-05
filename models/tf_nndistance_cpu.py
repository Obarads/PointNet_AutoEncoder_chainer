# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
cite charlesq34/pointnet-autoencoder
website:https://github.com/charlesq34/pointnet-autoencoder/blob/master/tf_ops/nn_distance/tf_nndistance_cpu.py
(access:2018/12/5 (yyyy/mm/dd))
"""

def nn_distance_cpu(pc1, pc2):
    '''
    Input:
        pc1: float TF tensor in shape (B,N,C) the first point cloud
        pc2: float TF tensor in shape (B,M,C) the second point cloud
    Output:
        dist1: float TF tensor in shape (B,N) distance from first to second 
        idx1: int32 TF tensor in shape (B,N) nearest neighbor from first to second
        dist2: float TF tensor in shape (B,M) distance from second to first 
        idx2: int32 TF tensor in shape (B,M) nearest neighbor from second to first
    '''
    N = pc1.get_shape()[1].value
    M = pc2.get_shape()[1].value
    pc1_expand_tile = tf.tile(tf.expand_dims(pc1,2), [1,1,M,1])
    #行列のn列目に1次元挿入
    #各次元を指定配列分だけ繰り返す、上の場合、２次元の部分をM回繰り返している。
    pc2_expand_tile = tf.tile(tf.expand_dims(pc2,1), [1,N,1,1])
    #pc1_expand_tile shape = pc2_expand_tile shape
    pc_diff = pc1_expand_tile - pc2_expand_tile # B,N,M,C
    #pc_diff is difference between pc1 and pc2 in coordinate system.
    pc_dist = tf.reduce_sum(pc_diff ** 2, axis=-1) # B,N,M
    dist1 = tf.reduce_min(pc_dist, axis=2) # B,N
    idx1 = tf.argmin(pc_dist, axis=2) # B,N
    dist2 = tf.reduce_min(pc_dist, axis=1) # B,M
    idx2 = tf.argmin(pc_dist, axis=1) # B,M
    return dist1, idx1, dist2, idx2


def verify_nn_distance_cup():
    np.random.seed(0)
    sess = tf.Session()
    pc1arr = np.random.random((1,5,3))
    pc2arr = np.random.random((1,5,3))
    pc1 = tf.constant(pc1arr)
    pc2 = tf.constant(pc2arr)
    """
    print("pc1arr:{}".format(pc1arr))
    print("pc2arr:{}".format(pc2arr))
    print("pc1:{}".format(pc1))
    print("pc2:{}".format(pc2))
    """
    dist1, idx1, dist2, idx2 = nn_distance_cpu(pc1, pc2)
    #print("dist1 sess.run:{}".format(sess.run(dist1)))
    #print("dist1:{}".format(dist1))
    #print(sess.run(idx1))
    #print("dist2 sess.run:{}".format(sess.run(dist2)))
    #print("dist2:{}".format(dist2))
    #print(sess.run(idx2))

    print(sess.run(dist1+dist2))
    print(sess.run(tf.reduce_mean(dist1+dist2)))

    """
    dist = np.zeros((5,6))
    for i in range(5):
        for j in range(6):
            dist[i,j] = np.sum((pc1arr[0,i,:] - pc2arr[0,j,:]) ** 2)
            
    print("dist")
    print(dist)
    """

if __name__ == '__main__':
    verify_nn_distance_cup()
