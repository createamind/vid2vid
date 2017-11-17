import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import ntpath
#from util.visualizer import Visualizer
#from util import html
import skvideo.io
import numpy as np
from PIL import Image
import cv2

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
#visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
#webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

def save_videos(web_dir, visuals, vid_path):
    vid_dir = os.path.join(web_dir, 'videos')
    short_path = ntpath.basename(vid_path[0])
    name = os.path.splitext(short_path)[0]


    for label, vid_numpy in visuals.items():
        vid_name = '%s_%s' % (name, label)
        #print(vid_name)
        save_path = os.path.join(vid_dir, vid_name)

        #print(vid_numpy.shape)

        for i in range(vid_numpy.shape[0]):
            #print(vid_numpy[i].shape)
            #cv2.imwrite(save_path + "_" + str(i) + ".png", vid_numpy[i])
            im = Image.fromarray(vid_numpy[i])
            im.save(save_path + "_" + str(i) + ".png")
        #for i in range(len(vid_numpy)):


'''
        writer = skvideo.io.FFmpegWriter(save_path, list(vid_numpy.shape))
        for i in xrange(list(vid_numpy.shape)[0]):
            writer.writeFrame(vid_numpy[i, :, :, :])
        writer.close()
'''



for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    vid_path = model.get_image_paths()
    #print(visuals)
    print('process video... %s' % vid_path)
    save_videos(web_dir, visuals, vid_path)
