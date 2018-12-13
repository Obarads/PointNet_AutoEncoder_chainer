'''
    Dataset for shapenet part segmentaion.
'''
# -*- coding: utf-8 -*-

import os
import os.path
import json
import numpy as np
import sys
import chainer
import provider
import h5py
import h5ed as ed
from distutils.util import strtobool
import argparse

def download_dataset():
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(BASE_DIR)
  # Download dataset for point cloud classification
  DATA_DIR = os.path.join(BASE_DIR, 'data')
  if not os.path.exists(DATA_DIR):
      os.mkdir(DATA_DIR)
  if not os.path.exists(os.path.join(DATA_DIR, 'shapenetcore_partanno_segmentation_benchmark_v0')):
      www = 'https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip'
      zipfile = os.path.basename(www)
      os.system('wget %s; unzip %s' % (www, zipfile))
      os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
      os.system('rm %s' % (zipfile))
  return DATA_DIR

#segmentation is unsupported.
class ChainerPointCloudDatasetH5(chainer.dataset.DatasetMixin):
    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/shapenetcore_h5/chair_train.h5'), augment=False):
        self.root = root
        self.augment = augment
        self.lenght = -1
        self.class_name = {}
        self.class_number = {}
        
        data = ed.decoding_hdf5_to_data(file_name=self.root)
        for d,i in zip(data,range(len(data))):
            if i == 0:
                self.data = np.array(data[d])
                self.label = np.full(len(data[d]),i,dtype=int)
            else:
                self.data = np.append(self.data, data[d], axis=0)
                self.label = np.append(self.label, np.full(len(data[d]),i,dtype=int), axis=0)

            self.class_name[i] = d
            self.class_number[d] = i        

        self.lenght = len(self.data)
        #variable_check(self)

    def __len__(self):
        return self.lenght

    def get_example(self, i):
        if self.augment:
            rotated_data = provider.rotate_point_cloud(
                self.data[i:i + 1, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            point_data = jittered_data[0]
        else:
            point_data = self.data[i]
        point_data = np.transpose(
            point_data.astype(np.float32), (1, 0))[:, :, None]
        return point_data, self.label[i]

    def get_data(self, i):
        return self.data[i]
        
    def get_label(self, i):
        return self.label[i]
    
    def get_data_array(self):
        return self.data

#segmentation is unsupported.
class ChainerPointCloudDatasetPCD(chainer.dataset.DatasetMixin):
    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/clustring_leg_1'), name_pattern='o_lrf_$.pcd',augment=False, num_point=70):
        self.root = root
        self.name_pattern = name_pattern
        self.augment = augment
        self.num_point = num_point
        self.lenght = -1
        self.class_name = {}
        self.class_number = {}
        #self.label = np.array()
        #self.data = np.array()

        import open3d.open3d as open3d
        if(os.path.isdir(self.root)):
            name_pattern_split = self.name_pattern.split("$")
            if(len(name_pattern_split)):
                name_pattern_front = name_pattern_split[0]
                name_pattern_back = name_pattern_split[1]
                file_search_sw = 1
                file_number = 0
                while file_search_sw:
                    file_path = os.path.join(self.root, name_pattern_front + str(file_number) + name_pattern_back)
                    if(os.path.isfile(file_path)):
                        pc = np.asarray(open3d.read_point_cloud(file_path).points)
                        choice = np.random.choice(len(pc), self.num_point, replace=True)
                        pc = pc[choice, :]
                        if file_number == 0:
                            self.data = np.array([pc])
                            self.label = np.array([0])
                        else:
                            self.data = np.append(self.data, np.array([pc]),axis=0)
                            self.label = np.append(self.label,np.array([0]),axis=0)
                        file_number+=1
                    else:
                        file_search_sw = 0

                self.data = pc_normalize(self.data)
                self.lenght = len(self.data)
        #variable_check(self)
    
    def __len__(self):
        return self.lenght

    def get_example(self, i):
        if self.augment:
            rotated_data = provider.rotate_point_cloud(
                self.data[i:i + 1, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            point_data = jittered_data[0]
        else:
            point_data = self.data[i]
        point_data = np.transpose(
            point_data.astype(np.float32), (1, 0))[:, :, None]
        return point_data, self.label[i]

    def get_data(self, i):
        return self.data[i]
        
    def get_label(self, i):
        return self.label[i]
    
    def get_data_array(self):
        return self.data


class ChainerPointCloudDatasetDefault(chainer.dataset.DatasetMixin):
    def __init__(self, root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/shapenetcore_partanno_segmentation_benchmark_v0'), 
    num_point=1024, classification=True, class_choice=None, split='train', normalize=True, augment=False):
        self.root = root
        self.num_point = num_point
        self.classification = classification
        self.class_choice = class_choice
        self.split = split
        self.normalize = normalize
        self.augment = augment
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.lenght = -1
        self.class_name = {}
        self.class_number = {}

        #allocate data directory divided by classes to self.cat 
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if self.class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in self.class_choice}

        self.meta = {}
        #allocate files name except extension to XX_ids
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

        #allocate file length
        count_label = 0
        # allocate each class length
        clasees_length = {}
        #print(self.cat)
        #example:{'Car': '02958343', 'Guitar': '03467517'} number is folder mame.
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            #get filename in points folder
            fns = sorted(os.listdir(dir_point))
            if self.split == 'trainval':
                fns = [fn for fn in fns if (
                    (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif self.split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif self.split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif self.split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (self.split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(
                    (os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
            #add [class lenght, lebel number]
            clasees_length[item] = len(self.meta[item])
            self.lenght += len(self.meta[item])
            self.class_name[count_label] = item
            self.class_number[item] = count_label
            count_label+=1

        #現在は座標のみとなっている。3のこと
        #self.dataにはすべてのファイルの点群データが読み込まれる。
        #予定図:[ファイル][点群][座標]
        self.data = np.zeros(shape=(self.lenght,self.num_point,3),dtype=float)
        #self.labelはlabelデータ
        if self.classification:
            self.label = np.zeros(shape=(self.lenght),dtype=int)
        else:
            self.label = np.zeros(shape=(self.lenght,self.num_point),dtype=int)
        #allocate number to label and data
        allocation_number = 0
        for item in self.cat:
            for n in range(clasees_length[item]):
                fp = self.meta[item][n]
                #extract point set from a pts file
                point_set = np.loadtxt(fp[0]).astype(np.float32)
                #nomalize
                if self.normalize:
                    point_set = pc_normalize(point_set)
                #num_point
                seg = np.loadtxt(fp[1]).astype(np.int64) - 1
                assert len(point_set) == len(seg)
                choice = np.random.choice(len(seg), self.num_point, replace=True)
                # resample
                point_set = point_set[choice, :]
                #allocate points
                self.data[allocation_number] = point_set
                #allocate label
                if self.classification:
                    self.label[allocation_number] = self.class_number[item]
                else:
                    self.label[allocation_number] = seg[choice]
                allocation_number += 1
        #メモリ対策?
        del self.meta
        #variable_check(self)

    def __len__(self):
        return self.lenght

    def get_example(self, i):
        if self.augment:
            rotated_data = provider.rotate_point_cloud(
                self.data[i:i + 1, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            point_data = jittered_data[0]
        else:
            point_data = self.data[i]
        point_data = np.transpose(
            point_data.astype(np.float32), (1, 0))[:, :, None]
        return point_data, self.label[i]

    def get_data(self, i):
        return self.data[i]
        
    def get_label(self, i):
        return self.label[i]
    
    def get_data_array(self):
        return self.data

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def variable_check(self):
    #self.data is input values.
    print(self.data)
    print(self.data.shape)
    #self.label is labels.
    print(self.label)
    print(self.label.shape)
    #self.class_number convert class name to class number.
    print(self.class_number)
    print(len(self.class_number))
    #self.class_number convert class number to class name.
    print(self.class_name)
    print(len(self.class_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='test_dataset')
    parser.add_argument('--type', '-c', type=str, default=None)
    parser.add_argument('--download', '-d', type=strtobool, default='false')
    args=parser.parse_args()

    type = args.type
    download = args.download
    #print(os.path.dirname(os.path.abspath(__file__)))
    if download:
        download_dataset()

    if type == 'h5':
        d = ChainerPointCloudDatasetH5()
        import utils.show3d_balls as show3d_balls
        show3d_balls.showpoints(d.get_data(0), ballradius=8)
    elif type == 'pcd':
        d = ChainerPointCloudDatasetPCD()
        import utils.show3d_balls as show3d_balls
        show3d_balls.showpoints(d.get_data(0), ballradius=8)
    elif type == 'default':
        d = ChainerPointCloudDatasetDefault(class_choice=["Chair"])
        #d = ChainerPointCloudDatasetDefault(class_choice=["Guitar","Car"])
        points, label = d[0]
        #print(points, label)
        import utils.show3d_balls as show3d_balls
        show3d_balls.showpoints(d.get_data(0), ballradius=8)