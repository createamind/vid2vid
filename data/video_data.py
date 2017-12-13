import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
import glob
import numpy as np
from collections import namedtuple,OrderedDict


server_ports = range(5550, 5558, 2)
from data.server import client


def make_dataset(data_path):
    file_list = glob.glob(os.path.join(data_path,'*.npy'))
    return file_list
def get_sensor(A,sensor_type = ["focus","angle"]):
    if sensor_type is not list:
        sensor_type = sensor_type.split(',')
    sensor_dict = { "focus":(0,5),"angle":(5,6),"track":(6,25),"trackPos":(25,26),"speedX":(26,27),"speedY":(27,28),"speedZ":(28,29),\
                    "wheelSpinVel":(29,33),"rpm":(33,34),"speedXDelta":(34,35),"angleDelta":(35,36),"trackPosDelta":(36,37),\
                    "damage":(37,38),"opponents":(38,74),"action":(74,76),"reward":(76,)
                    }
    Obs = OrderedDict()


    try :

        [ setattr(Obs,i , A[:,:,list(range(*sensor_dict[i]))]) for i in sensor_type]
    except Exception as e:
        print("sensor_type not in ",sensor_dict)
        raise e
    return Obs

class VideoDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
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
        filename, ABS= next(self.c)
        AB = ABS[0]
        print("AB",AB.shape)
        if  self.opt.sensor_types:
            S = ABS[1]


            sensor_data = get_sensor(S,self.opt.sensor_types)
            # print("sensor data",sensor_data.action,sensor_data.action.shape)
        #     sensor  (2, 24, 82)


        AB_path = filename
        A = AB[:,0:self.opt.input_nc]/127.5 -1.

        B = AB[:,self.opt.input_nc:]/127.5 -1.
        # print("====== load A size ==== {0}".format(A.shape))
        # print("====== load B size ==== {0}".format(B.shape))
        return {'A': A, 'B': B,"speedX":sensor_data.speedX,
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

