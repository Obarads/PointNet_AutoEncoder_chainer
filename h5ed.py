import os
import os.path
import sys
import h5py
import argparse
import chainer_dataset as pd
from distutils.util import strtobool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def encoding_data_to_hdf5(data,file_name="point_data.h5",keys=None):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset(keys, data=data)
        f.flush()
        f.close()

def decoding_hdf5_to_data(file_name="point_data.h5"):
    data = {}
    with h5py.File(file_name, 'r') as f:
        for key in f.keys():
            data[key] = f[key].value
    return data

def main():
    parser = argparse.ArgumentParser(
        description='encoding_data_to_hdf5')
    parser.add_argument('--class_choice', '-c', type=str, default='Chair')
    parser.add_argument('--file_name', '-f', type=str, default='point_data.h5')
    parser.add_argument('--data_path', '-d', type=str, default=os.path.join(
        BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0'))
    parser.add_argument('--test_d', '-td', type=strtobool, default='false')
    parser.add_argument('--test_e', '-te', type=strtobool, default='true')
    args = parser.parse_args()

    class_choice = args.class_choice
    file_name = args.file_name
    data_path = args.data_path
    test_d = args.test_d
    test_e = args.test_e
    if test_e:
        d = pd.ChainerPointCloudDatasetDefault(root=data_path,class_choice=class_choice)
        encoding_data_to_hdf5(data=d.get_data_array(), file_name=file_name,keys=class_choice)

    if test_d:
        data = decoding_hdf5_to_data(file_name=file_name)
        import utils.show3d_balls as show3d_balls
        show3d_balls.showpoints(data[class_choice][0], ballradius=8)


if __name__ == '__main__':
    main()