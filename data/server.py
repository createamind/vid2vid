"""A Socket subclass that adds some serialization methods."""

import zlib
import pickle
import glob
import numpy as np

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
    def send_array_(self, A, flags = 0, copy = True, track = False, filename = None):
        #print('ffff',filename)
        self.send_json(filename , flags | zmq.SNDMORE)
        self.send_array( A, flags = flags, copy = copy, track = track)

    def recv_array_(self, flags = 0, copy = True, track = False):

        filename = self.recv_json(flags = flags)
        return filename , self.recv_array(flags = flags, copy = copy, track = track)


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
                [s.send_array_(data.copy(order='C'), copy=False, filename=data_path) for data in gen]
            except StopIteration:
                data_path, gen = video_data_gen(random.choice(f_lst), opt)
        else:
            data_path, gen = data_gen(random.choice(f_lst), skip = opt.skip, length = opt.depth, pre = opt.depth)
            try:
                [s.send_array_(data.copy(order='C'), copy=False, filename=data_path) for data in gen]
            except StopIteration:
                data_path, gen = data_gen(random.choice(f_lst), skip = opt.skip, length = opt.depth, pre = opt.depth)

def client(opt = None):
    hwm = 20
    host = 'localhost'
    server_ports = range(int(5550 + 10*opt.depth), int(5558 + 10*opt.depth))
    setup_server(server_ports,opt)
    ctx = SerializingContext()

    c = ctx.socket(zmq.PULL)
    c.set_hwm(hwm)
    [c.connect('tcp://{}:{}'.format(host,p)) for p in server_ports]
    res = []
    while 1:
        filename , a = c.recv_array_(copy = False)
        array = [ c.recv_array_(copy = False)[1]for i in range(opt.batchSize) ]
        #print(filename)
        A = np.concatenate(array,axis = 0)
        #print('gen',A.shape)
        yield filename , A

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