"""A Socket subclass that adds some serialization methods."""

import zlib
import pickle
import glob
import numpy as np
from collections import namedtuple,OrderedDict
import zmq
import random
import logging

from options.train_options import TrainOptions
from multiprocessing import Process
from data.img_loder import data_gen, video_data_gen

logger = logging.getLogger(__name__)

opt = TrainOptions().parse()
## f_lst = glob.glob(opt.data_dir + 'v_BabyCrawling**.avi')
f_lst = glob.glob(opt.data_dir)
##f_lst = glob.glob('/data/dataset/depthdata/vkitti_1.3.1_rgb/**/**/')
print("Total videos: {}".format(len(f_lst)))


class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods

    send_zipped_pickle is just like send_pyobj, but uses
    zlib to compress the stream before sending.

    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    """

    def send_zipped_pickle(self, obj, flags = 0, protocol = -1):
        """pack and compress an object with pickle and zlib."""
        pobj = pickle.dumps(obj, protocol)
        zobj = zlib.compress(pobj)
        print('zipped pickle is %i bytes' % len(zobj))
        return self.send(zobj, flags = flags)

    def recv_zipped_pickle(self, flags = 0):
        """reconstruct a Python object sent with zipped_pickle"""
        zobj = self.recv(flags)
        pobj = zlib.decompress(zobj)
        return pickle.loads(pobj)

    def send_array(self, A, flags = 0, copy = True, track = False):
        """send a numpy array with metadata"""
        md = dict(
            dtype = str(A.dtype),
            shape = A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy = copy, track = track)

    def recv_array(self, flags = 0, copy = True, track = False):
        """recv a numpy array"""
        md = self.recv_json(flags = flags)
        msg = self.recv(flags = flags, copy = copy, track = track)
        A = np.frombuffer(msg, dtype = md['dtype'])
        return A.reshape(md['shape'])

class imgsocket(SerializingSocket):
    def send_array_(self, A, flags = 0, copy = True, track = False, filename = None,num =1):
        #print('ffff',filename)
        self.send_json(filename , flags | zmq.SNDMORE)
        if num == 1:
            self.send_array( A.copy(order='C'), flags = flags, copy = copy, track = track)
        if num >1:
            [ self.send_array(A[i].copy(order = 'C'), flags = flags |  zmq.SNDMORE if i != num-1 else flags , copy = copy, track = track) for i in range(num)]



    def recv_array_(self, flags = 0, copy = True, track = False,num =1):

        filename = self.recv_json(flags = flags)
        if num == 1:
            data = self.recv_array(flags = flags, copy = copy, track = track)
        if num >1:
            data = [ self.recv_array(flags = flags, copy = copy, track = track) for i in range(num)]

        return filename , data


class SerializingContext(zmq.Context):
    _socket_class = imgsocket


def test():
    hwm = 20
    ctx = SerializingContext()
    #client
    pull = ctx.socket(zmq.PULL)
    pull.set_hwm(hwm)
    #server
    push = ctx.socket(zmq.PUSH)
    push.set_hwm(hwm)

    #set port
    push.bind('tcp://*:6666')
    pull.connect('tcp://localhost:6666')

    A = np.ones((3, 3))
    print("Array is %i bytes" % (A.nbytes))

    # send/recv with pickle+zip
    push.send_zipped_pickle(A)
    B = pull.recv_zipped_pickle()
    # now try non-copying version
    push.send_array(A, copy = False)
    C = pull.recv_array(copy = False)
    logger.info('client started'*34)
    print("Checking zipped pickle...")
    print("Okay" if (A == B).all() else "Failed")
    print("Checking send_array...")
    print("Okay" if (C == B).all() else "Failed")


def start_server(port , opt ):
    hwm = 20
    ctx = SerializingContext()

    s = ctx.socket(zmq.PUSH)
    s.set_hwm(hwm)


    s.bind('tcp://*:{}'.format(port))


    # fix ndarray not continious bug :  array.copy(order='C')

    while 1:
        if opt.load_video == 1:
            data_path, gen = video_data_gen(random.choice(f_lst), opt )
            #print(data_path, gen)
            try:
                [s.send_array_(data, copy=False, filename=data_path) for data in gen]
            except StopIteration:
                data_path, gen = video_data_gen(random.choice(f_lst), opt)
        else:
            data_path, gen = data_gen(random.choice(f_lst), opt)
            try:
                [s.send_array_(data, copy=False, filename=data_path,num = opt.input_num) for data in gen]
            except StopIteration:
                data_path, gen = data_gen(random.choice(f_lst), opt)

def client(opt = None):
    hwm = 20
    host = 'localhost'
    server_ports = range(int(6550 + 0*opt.depth), int(6550 + 1*opt.depth))
    setup_server(server_ports,opt)
    ctx = SerializingContext()

    c = ctx.socket(zmq.PULL)
    c.set_hwm(hwm)
    [c.connect('tcp://{}:{}'.format(host,p)) for p in server_ports]
    res = []
    while 1:


        filename , a = c.recv_array_(copy = False,num = opt.input_num)
        all_array = [ c.recv_array_(copy = False,num = opt.input_num) for i in range(opt.batchSize) ]
        rbg_array = [i[1][0] for i in all_array]

        if opt.sensor_types:
            s_array = [i[1][1] for i in all_array]
            S = np.concatenate(s_array, axis = 0)
            A = np.concatenate(rbg_array, axis = 0)
        else:
            A = np.concatenate([rbg_array], axis = 0)

        #print(filename)



        # print("rbg_array .shape "*2,rbg_array[0].shape)
        # print("A .shape " * 2, A.shape)
        # print("rec S "*22,S)
        if not opt.sensor_types:
            yield filename , [A ]
        else :
            yield filename , [A ,S]

'''
load_video = 1
vid_root = '/data/dataset/UCF/'
vid_ls = glob.glob(vid_root+"v_BabyCrawling**.avi")
f_lst = glob.glob('/data/dataset/depthdata/vkitti_1.3.1_rgb/**/**/')
'''


def setup_server(server_ports, opt):
    # Now we can run a few servers
    print("Server starts ...")
    for p in server_ports:
        Process(target = start_server, args = (p ,opt )).start()

    # Now we can connect a client to all these servers
    #Process(target = client, kwargs = {'ports' : server_ports}).start()






if __name__=="__main__":

    Opt = namedtuple("opt",[ "depth","batchSize","load_video","skip","overlap","data_root","input_nc","output_nc" ,"pre","input_num","sensor_types"])
    opt = Opt(depth = 2,batchSize = 2,load_video = 0,skip = 1,overlap =0,data_root = "/data/dataset/torcs_data/**/",input_nc = 3,output_nc = 3,pre = 2,input_num =2,sensor_types = "angle,speedX,speedY")

    f_lst = glob.glob(opt.data_root)
    c = client(opt)

    for i, d in c:
        print(type(d))
        print('d[0]',d[0][0])
        print('d[1]', d[0][1])


"""
        self._ob_status = np.hstack((_focusScaled,  5 , 0-4
                                     status.angle / np.pi, 1 ,5
                                     _track,                19 ,6-24
                                     status.trackPos,        1,25
                                     status.speedX / 200.,   1,26
                                     status.speedY / 200.,   1,27
                                     status.speedZ / 200.,   1,28
                                     np.array(status.wheelSpinVel) / 100.0,  4, 29-32
                                     status.rpm / 10000.,  1,33
                                     speedXDelta,          1,34
                                     angleDelta * 10.,     1,35
                                     trackPosDelta * 10.,  1,36
                                     status.damage,        1,37
                                     np.array(status.opponents) / 200.,  36, 38-73
                                     

                                     ))



"""



