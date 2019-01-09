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
first, train model with lrf_data set.
python train.py -p data/lrf_data/torso_train/ --num_point=70 --extension=pcd -g 0

second, execute other_test.py
python other_test.py --num_point=70 -lf result/model.npz -p torso
"""

def HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/torso_train'),file_name_pattern='o_lrf_$.pcd',method=1):

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
                    p1 = np.array([pc[0][0],pc[0][1]])
                    pm = np.array([pc[len(pc)-1][0],pc[len(pc)-1][1]])

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
                    center = np.zeros(2)
                    L1[0] = pm[0] - p1[0]
                    L1[1] = pm[1] - p1[1]
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
                            center[0] = dp[0]
                            center[1] = dp[1]
                    if method == 2:
                        v1 = p1 - center
                        v2 = pm - center
                        cos = (v1[0]*v2[0]+v1[1]*v2[1])/(np.sqrt(v1[0]**2+v1[1]**2)*np.sqrt(v2[0]**2+v2[1]**2))
                        angle = np.arccos(cos)
                        w_g = width/grith
                        results.append([width, grith, depth_max, w_g, angle])
                        if "torso_train" in path and depth >= 0.07:
                            print([width, grith, depth_max, w_g, angle])
                    else:
                        results.append([width, grith, depth_max])
                    file_number+=1
                else:
                    file_search_sw = False

    results = np.array(results)

    return results

def main():
    parser = argparse.ArgumentParser(
        description='AutoEncoder ShapeNet')
    parser.add_argument('--trans', '-t',type=strtobool, default='true')
    parser.add_argument('--residual', '-r',type=strtobool, default='false')
    parser.add_argument('--load_file', '-lf', type=str, default='result/model.npz')
    parser.add_argument('--num_point', '-np', type=int, default=50)
    parser.add_argument('--parts', '-p',type=str, default='hip')
    parser.add_argument('--svm_method', '-sm',type=int, default=1)
    args = parser.parse_args()

    trans = args.trans
    residual = args.residual
    load_file = args.load_file
    num_point = args.num_point
    parts = args.parts
    svm_method = args.svm_method

    trans_lam1 = 0.001
    trans_lam2 = 0.001
    out_dim = 3
    in_dim = 3
    middle_dim = 64
    dropout_ratio = 0
    use_bn = True

    #Learned 3D Feature
    print('Load PointNet-AutoEncoder model... load_file={}'.format(load_file))
    model = ae.PointNetAE(out_dim=out_dim, in_dim=in_dim, middle_dim=middle_dim, dropout_ratio=dropout_ratio, use_bn=use_bn,
                          trans=trans, trans_lam1=trans_lam1, trans_lam2=trans_lam2, residual=residual,output_points=num_point)
    serializers.load_npz(load_file, model)
    """
    ae_train = dataset.convert_pcd_to_array(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/'+parts+'_train'),file_name_pattern='o_lrf_$.pcd',num_point=num_point)
    ae_train = dataset.ChainerPointCloudDataset(ae_train,np.zeros(len(ae_train)))
    ae_test_t = dataset.convert_pcd_to_array(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/'+parts+'_test'),file_name_pattern='o_lrf_$.pcd',num_point=num_point)
    ae_test_t = dataset.ChainerPointCloudDataset(ae_test_t,np.zeros(len(ae_test_t)))

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

    #test -true
    t_test_output = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in range(len(ae_test_t)):
            x,_ = ae_test_t.get_example(i)
            x = chainer.Variable(np.array([x]))
            h,_,_=model.encoder(x)
            t_test_output.append(h.array)
        t_test_output = np.array(t_test_output)
    dn,_,df,_,_= t_test_output.shape
    t_test_output = np.reshape(t_test_output,[dn,df])
    ae_result_t = clf_one.predict(t_test_output)
    t_count = 0
    for (i,r) in enumerate(ae_result_t):
        if r == 1:
            t_count+=1
    ae_acc_t = t_count/len(ae_result_t)

    #test -false
    f_test_output = []
    ae_test_f = dataset.convert_pcd_to_array(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/not_'+parts),file_name_pattern='o_lrf_$.pcd',num_point=num_point)
    ae_test_f = dataset.ChainerPointCloudDataset(ae_test_f,np.zeros(len(ae_test_f)))
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in range(len(ae_test_f)):
            x,_ = ae_test_f.get_example(i)
            x = chainer.Variable(np.array([x]))
            h,_,_=model.encoder(x)
            f_test_output.append(h.array)
        f_test_output = np.array(f_test_output)
    dn,_,df,_,_= f_test_output.shape
    f_test_output = np.reshape(f_test_output,[dn,df])
    ae_result_f = clf_one.predict(f_test_output)
    f_count = 0
    for r in ae_result_f:
        if r == -1:
            f_count+=1
    ae_acc_f = f_count/len(ae_result_f)
    ae_acc_total = (f_count+t_count)/(len(ae_result_f)+len(ae_result_t))
    print("ae_acc_t:{}% ".format(ae_acc_t*100))
    print("ae_acc_f:{}% ".format(ae_acc_f*100))
    print("ae_acc_total:{}% ".format(ae_acc_total*100))
    """

    #Hand-crafted 3D Feature
    import csv

    svm_train = HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/'+parts+'_train'),method=svm_method)
    print(svm_train)
    svm_test_t = HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/'+parts+'_test'),method=svm_method)
    svm_test_f = HandCrafted3DFeature(path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/lrf_data/not_'+parts),method=svm_method)

    clf_two = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma="auto")
    clf_two.fit(svm_train)

    #test -true
    svm_result_t = clf_two.predict(svm_test_t)
    svm_t_count = 0
    for r in svm_result_t:
        if r == 1:
            svm_t_count+=1
    svm_acc_t = svm_t_count/len(svm_result_t)
    #test -false
    svm_result_f = clf_two.predict(svm_test_f)
    svm_f_count = 0
    for r in svm_result_f:
        if r == -1:
            svm_f_count+=1
    svm_acc_f = svm_f_count/len(svm_result_f)    
    svm_acc_total = (svm_f_count+svm_t_count)/(len(svm_result_f)+len(svm_result_t))
    print("svm_acc_t:{}% ".format(svm_acc_t*100))
    print("svm_acc_f:{}% ".format(svm_acc_f*100))
    print("svm_acc_total:{}% ".format(svm_acc_total*100))

if __name__ == '__main__':
    main()