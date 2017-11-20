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
    name = ntpath.basename(vid_path).split('.')[0]
    #print("vid_dir: {}".format(vid_dir))
    #print("name: {}".format(name))

    vid_numpy = np.concatenate((visuals['real_A'], visuals['real_B'], visuals['fake_B']), axis=2)
    #print(vid_numpy.shape)

    for i in range(vid_numpy.shape[0]):
        save_path = vid_dir
        save_name = name + '_' + str(i) +'.png'
        print("save_path: {}".format(save_path+save_name))

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        img = vid_numpy[i][:, :, ::-1]
        #print(img.shape)
        cv2.imwrite(save_path+save_name, img)


for i, data in enumerate(dataset):
    if i >= opt.how_many:
        print('break')
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    vid_path = model.get_image_paths()
    #print(visuals)
    print('process video... %s' % vid_path)
    save_videos(web_dir, visuals, vid_path)
