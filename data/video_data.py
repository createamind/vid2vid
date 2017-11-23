import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
import glob
import numpy as np
from collections import namedtuple


server_ports = range(5550, 5558, 2)
from data.server import client


def make_dataset(data_path):
    file_list = glob.glob(os.path.join(data_path,'*.npy'))
    return file_list

class VideoDataset(BaseDataset):

    def initialize(self, opt):
        #opt = namedtuple('option', ['dataroot', 'phase'])
        #opt.dataroot = '.'
        #opt.phase = 'data'
        
        #self.opt = opt
        #self.root = opt.dataroot
        #self.data_path = os.path.join(opt.dataroot, opt.phase)
        self.data_list = make_dataset(opt.dataroot)
        self.c = client(opt = opt)
        self.max_size = opt.max_dataset_size
        #print(self.data_list)

        #transform_list = [transforms.ToTensor()]
        #self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        if index >= self.max_size:
            raise IndexError

        #AB = np.load(AB_path)
        filename, AB = next(self.c)
        AB_path = filename
        A = AB[::2]/127.5 -1.
        #print("====== load A size ==== {0}".format(A.shape))
        B = AB[1::2]/127.5 -1.
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'VideoDataset'

if __name__ == '__main__':
    opt = namedtuple('option',['dataroot'])
    opt.dataroot = './data/'
    v = VideoDataset()
    v.initialize(opt)
    for i in range(10):
        a = np.ones([2,3,7,256,256], dtype=np.float32)
        np.save('data/{}.npy'.format(i),a)
    for i in v:
        print(i['A'].shape)

