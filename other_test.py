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
from sklearn import svm

import numpy as np
import os
import argparse
from distutils.util import strtobool

# self made
import models.pointnet_ae as ae
import dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
# First, convert pcd datas to h5 files.
#torso
python dataset.py -p data/lrf_data/verified_0_torso_1/ -r o_lrf_$.pcd -n 50 -k torso --h5_name torso_1.h5
python dataset.py -p data/lrf_data/verified_0_torso_2/ -r o_lrf_$.pcd -n 50 -k torso --h5_name torso_2.h5
#leg
python dataset.py -p data/lrf_data/verified_0_leg_1/ -r o_lrf_$.pcd -n 50 -k torso --h5_name leg_1.h5
python dataset.py -p data/lrf_data/verified_0_leg_2/ -r o_lrf_$.pcd -n 50 -k torso --h5_name leg_2.h5
#hip
python dataset.py -p data/lrf_data/verified_0_hip_1/ -r o_lrf_$.pcd -n 50 -k torso --h5_name hip_1.h5
python dataset.py -p data/lrf_data/verified_0_hip_2/ -r o_lrf_$.pcd -n 50 -k torso --h5_name hip_2.h5

# Second, 
"""


def HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/verified_0_leg_1'),file_name_pattern='o_lrf_$.pcd'):

    import open3d.open3d as open3d
    if(os.path.isdir(path)):
        name_pattern_split = file_name_pattern.split("$")
        if(len(name_pattern_split)==2):
            name_pattern_front = name_pattern_split[0]
            name_pattern_back = name_pattern_split[1]
            file_search_sw = True
            file_number = 0
            #N, ch  ch=width, grith, depth
            results = []
            while file_search_sw:
                file_path = os.path.join(path, name_pattern_front + str(file_number) + name_pattern_back)
                if(os.path.isfile(file_path)):
                    pc = np.asarray(open3d.read_point_cloud(file_path).points)
                    p1 = pc[0]
                    pm = pc[len(pc)-1]

                    #width
                    width = np.sqrt((p1[0]-pm[0])**2+(p1[1]-pm[1])**2)

                    #grith
                    ago = 0
                    grith = 0
                    for dp,i in zip(pc,range(len(pc))):
                        if i == 0:
                            grith = 0
                            ago = dp
                        else:
                            grith += np.sqrt((dp[0]-ago[0])**2+(dp[1]-ago[1])**2)
                            ago = dp

                    #depth
                    #x,y
                    L1 = np.zeros(2)
                    depth_max = 0
                    L1[0] = p1[0] - pm[0]
                    L1[1] = p1[1] - pm[1]
                    L1TL1 = L1[0]**2 + L1[1]**2
                    for dp,i in zip(pc,range(len(pc))):
                        L2 = np.zeros(2)
                        L2[0] = dp[0] - p1[0]
                        L2[1] = dp[1] - p1[1]
                        L2TL2 = L2[0]**2 + L2[1]**2
                        L1TL2 = L1[0]*L2[0] + L1[1]*L2[1]
                        depth = (L2TL2 * L1TL1 - L1TL2**2)/L1TL1
                        if depth > depth_max:
                            depth_max = depth
                    results.append([width, grith, depth_max])
                    file_number+=1
                else:
                    file_search_sw = False

    results = np.array(results)

    return results

def main():
    parser = argparse.ArgumentParser(
        description='AutoEncoder ShapeNet')
    parser.add_argument('--trans', type=strtobool, default='true')
    parser.add_argument('--residual', type=strtobool, default='false')
    parser.add_argument('--load_file', '-lf', type=str, default='result/model.npz')
    parser.add_argument('--num_point', type=int, default=50)
    parser.add_argument('--parts', type=str, default='hip')
    parser.add_argument('--train_path', '-trp', type=str, default=None)
    parser.add_argument('--test_path', '-tep', type=str, default=None)
    args = parser.parse_args()

    trans = args.trans
    residual = args.residual
    load_file = args.load_file
    num_point = args.num_point
    parts = args.parts
    train_path = args.train_path
    test_path = args.test_path

    trans_lam1 = 0.001
    trans_lam2 = 0.001
    out_dim = 3
    in_dim = 3
    middle_dim = 64
    dropout_ratio = 0
    use_bn = False

    #Learned 3D Feature
    print('Load PointNet-AutoEncoder model... load_file={}'.format(load_file))
    model = ae.PointNetAE(out_dim=out_dim, in_dim=in_dim, middle_dim=middle_dim, dropout_ratio=dropout_ratio, use_bn=use_bn,
                          trans=trans, trans_lam1=trans_lam1, trans_lam2=trans_lam2, residual=residual,output_points=num_point)
    serializers.load_npz(load_file, model)

    ae_train = dataset.convert_h5_to_array(train_path)
    ae_train = dataset.ChainerPointCloudDataset(ae_train,np.zeros(len(ae_train)))
    ae_test = dataset.convert_h5_to_array(test_path)
    ae_test = dataset.ChainerPointCloudDataset(ae_test,np.zeros(len(ae_test)))

    #svm train
    output = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in range(len(ae_train)):
            x,_ = ae_train.get_example(i)
            x = chainer.Variable(np.array([x]))
            h,_,_=model.encoder(x)
            output.append(h.array)
        output = np.array(output)
    #print(output.shape)
    dn,_,df,_,_= output.shape
    output = np.reshape(output,[dn,df])
    clf_one = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma="auto")
    clf_one.fit(output)

    #test
    test_output = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in range(len(ae_test)):
            x,_ = ae_test.get_example(i)
            x = chainer.Variable(np.array([x]))
            h,_,_=model.encoder(x)
            test_output.append(h.array)
        test_output = np.array(test_output)
    dn,_,df,_,_= test_output.shape
    test_output = np.reshape(test_output,[dn,df])
    result = clf_one.predict(test_output)
    count = 0
    for r in result:
        if r == 1:
            count+=1
    acc = count/len(result)
    print("acc:{}%".format(acc*100))

    #Hand-crafted 3D Feature
    import csv

    svm_train = HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/verified_0_'+parts+'_1'))
    svm_test = HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/verified_0_'+parts+'_2'))

    clf_two = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma="auto")
    clf_two.fit(svm_train)

    result_2 = clf_two.predict(svm_test)
    count_2 = 0
    for r in result_2:
        if r == 1:
            count_2+=1
    acc_2 = count_2/len(result_2)
    print("acc:{}%".format(acc_2*100))

if __name__ == '__main__':
    main()