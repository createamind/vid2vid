#coding: utf-8
import numpy as np
import distutils.spawn
import distutils.version
import os
#import os.path as osp
import subprocess

#from datetime import datetime
#import time






class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise Exception(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(
                    frame_shape))
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise Exception(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend': self.backend,
            'version': str(subprocess.check_output(
                [self.backend, '-version'],
                stderr=subprocess.STDOUT)),
            'cmdline': self.cmdline
        }

    def start(self):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel', 'error',  # suppress warnings
            '-y',
            '-r', '%d' % self.frames_per_sec,
            '-f', 'rawvideo',  # input
            '-s:v', '{}x{}'.format(*self.wh),
            '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
            '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly
            '-vf', 'vflip',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            # '-threads',6,
            self.output_path
        )

        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise Exception("VideoRecorder encoder exited with status {}".format(ret))

def comine_arr(result):
    print('comb shape',result['real_A'].shape)
    real_A = result['real_A'][0]
    real_B = result['real_B'][0]
    fake_B = result['fake_B'][0]
    # if len(real_A.shape) ==5:
    #     print('real_A.shape',real_A.shape)
    #     real_A = real_A[0]
    #     real_B = real_B[0]
    #     fake_B = fake_B[0]
    # real_A= np.transpose(real_A[0], (1, 2, 3, 0))
    # real_B = np.transpose(real_B[0], (1, 2, 3, 0))
    # fake_B = np.transpose(fake_B[0], (1, 2, 3, 0))
    v = []
    big_arr = np.ones([256,256*3,3],np.float32)
    for a,b,c in zip(real_A,real_B,fake_B):
        big_arr[:,:256,:] = real_A
        big_arr[:, 256:256*2, :] = real_B
        big_arr[:, 256*2:, :] = fake_B
        v.append(big_arr)

    vs = np.asarray(v,dtype = np.float32)/255.
    print('final array',vs.max(),vs.min())
    return vs

def save_video(result,name,fps =24,dir_path= '/tmp'):
    frames = comine_arr(result)


    shape = frames[0].shape
    print('{}/{}.mp4'.format(dir_path,name))

    encoder = ImageEncoder('{}/{}.mp4'.format(dir_path,name), shape, fps)

    for frame in frames:
        encoder.capture_frame(frame)
    encoder.close()
if __name__ == '__main__':
    frames = None
    shape = frames[0].shape

    encoder = ImageEncoder('{}.mp4'.format('v'), shape, 24)

    for frame in frames:
        encoder.capture_frame(frame)
    encoder.close()

