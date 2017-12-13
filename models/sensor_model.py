import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
# from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.pix2pix_model import Pix2PixModel
from models.networks import UnetSkipConnectionBlock

class sensor_model(Pix2PixModel):
    def name(self):
        return "sensor_model"

    def initialize(self, opt):
        Pix2PixModel.initialize(self, opt)
        self.action = self.Tensor(opt.batchSize, opt.depth, 2)


    def set_input(self, input):
        Pix2PixModel.set_input(self, input)
        action = torch.from_numpy(input["action"])
        self.action.resize_(action.size()).copy_(action)
        if torch.cuda.is_available():
            self.action = self.action.cuda()
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B ,code= self.netG(self.real_A)
        print("code",code.size())

        self.real_B = Variable(self.input_B)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        print("A   B"*5,self.real_A.size(),self.fake_B.size())
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).data
        fake_AB_ = Variable(fake_AB)
        pred_fake = self.netD(fake_AB_.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        #action
        # self.action_loss = self.criterionL2(self.action,self.action_prediction)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 #+self.action_loss

        self.loss_G.backward()








