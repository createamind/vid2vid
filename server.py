"""A Socket subclass that adds some serialization methods."""

import zlib
import pickle

import numpy as np

import zmq

import logging
logger = logging.getLogger(__name__)

import threading
from multiprocessing import Pool , Process
from functools import partial
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


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


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


def start_server(port = '5557' ,hwm =20):
    ctx = SerializingContext()

    s = ctx.socket(zmq.PUSH)
    s.set_hwm(hwm)


    s.bind('tcp://*:{}'.format(port))
    A = np.ones((3, 3))+ port

    while 1:
        print('send', A)

        s.send_array(A, copy = False)

def client(host = 'localhost',ports = [5555],hwm =20):
    ctx = SerializingContext()

    c = ctx.socket(zmq.PULL)
    c.set_hwm(hwm)
    [c.connect('tcp://{}:{}'.format(host,p)) for p in ports]
    while 1:
        A = c.recv_array(copy = False)
        print('recv',A)





if __name__ == "__main__":
    # Now we can run a few servers
    server_ports = range(5550, 5558, 2)
    for p in server_ports:
        Process(target = start_server, kwargs = {'port' :p},).start()

    # Now we can connect a client to all these servers
    Process(target = client, kwargs = {'ports' : server_ports}).start()