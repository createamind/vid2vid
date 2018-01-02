from collections import OrderedDict

import torch
import numpy as np
from torch.autograd import Variable
from models.base_model import BaseModel
import models.networks as networks
import warnings
import util.util as util


class Vid2SeqModel(BaseModel):
    """
    Generate sequence conditioned on input video data, using GAN with CNN-LSTM generator and
        multi-layer bidirectional LSTM as discriminator
    """

    def name(self):
        return 'Vid2SeqModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.dataset_mode = opt.dataset_mode
        self.isTrain = opt.isTrain
        self.opt = opt
        # define tensors
        # 3D tensor shape (N,Cin,Din,Hin,Win)
        # self.input_vid = self.Tensor(opt.batchSize, opt.input_nc,
        #                          opt.depth, opt.fineSize, opt.fineSize)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   int(opt.depth / 2), opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   int(opt.depth / 2), opt.fineSize, opt.fineSize)

        self.action_A = self.Tensor(opt.batchSize, int(opt.depth / 2))
        self.action_B = self.Tensor(opt.batchSize, int(opt.depth / 2))


        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,  # of gen filters in first conv layer
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_seq = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD_seq,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_vid = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                              opt.which_model_netD_vid,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD_vid, 'D', opt.which_epoch)
                self.load_network(self.netD_seq, 'D', opt.which_epoch)

        if self.isTrain:
            # 3D Change
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_vid = torch.optim.Adam(self.netD_vid.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_seq = torch.optim.Adam(self.netD_seq.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_vid)
            self.optimizers.append(self.optimizer_D_seq)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_vid)
            networks.print_network(self.netD_seq)
        print('-----------------------------------------------')

    def set_input(self, inputs, opt):
        """
        :param is_numpy: using numpy array or not
        :param inputs: a dict contains two inputs forms with key: 'video' and 'target_seq'
        :return:
        """
        ## numpy to torch tensor

        AtoB = self.opt.which_direction == 'AtoB'
        A = inputs['A']
        inputs['A'] = np.split(A, 2, axis=2)[0]
        inputs['B'] = np.split(A, 2, axis=2)[1]

        input_A = torch.from_numpy(inputs['A' if AtoB else 'B']).float()
        # print("======input A SIZE==== {0}".format(input_A.size()))
        input_B = torch.from_numpy(inputs['B' if AtoB else 'A']).float()
        # speedX = torch.from_numpy(inputs["speedX"])  # with the length lX = lA + lB
        action_A = torch.from_numpy(np.split(inputs["action"], 2, axis=1)[0]).float()/5
        action_B = torch.from_numpy(np.split(inputs["action"], 2, axis=1)[1]).float()/5

        self.input_A = Variable(input_A)
        self.input_B = Variable(input_B)
        self.action_A = Variable(action_A)
        self.action_B = Variable(action_B)

        # convert to cuda
        if self.gpu_ids and torch.cuda.is_available():
            self.input_A = self.input_A.cuda()
            self.input_B = self.input_B.cuda()

            self.action_A = self.action_A.cuda()
            self.action_B = self.action_B.cuda()

        self.image_paths = inputs['A_paths' if AtoB else 'B_paths']
        self.real_A = self.input_A
        self.real_B = self.input_B

        if opt.debug :
            print(inputs["action"])
            print(self.action_A)
            print(self.action_B)

    def forward(self):
        self.fake_B, self.action_B_pred = self.netG(self.input_A)
        if self.opt.debug :
            print("." * 10 + "Compare sequences" + "." * 10)
            print(self.action_A.data)
            print(self.action_B_pred.data)
            print(self.action_B_pred)
            print("." * 10 + "Compare sequences" + "." * 10)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).data
        fake_AB_ = Variable(fake_AB)
        fake_cat_seq = torch.cat([self.action_A, self.action_B_pred], 1)
        pred_fake = self.netD_vid(fake_AB_.detach())
        action_fake = self.netD_seq(fake_cat_seq.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, False) + self.criterionGAN(action_fake, False)  # fake speed

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_cat_seq = torch.cat([self.action_A, self.action_B], 1)
        pred_real_vid = self.netD_vid(real_AB.detach())
        pred_real_seq = self.netD_seq(real_cat_seq.detach())
        self.loss_D_real = self.criterionGAN(pred_real_vid, True) + self.criterionGAN(pred_real_seq, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward(retain_graph=True)

    def backward_G(self):

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_vid_fake = self.netD_vid(fake_AB)
        fake_cat_seq = torch.cat([self.action_A, self.action_B_pred], 1)
        pred_action_fake = self.netD_seq(fake_cat_seq)
        #self.loss_G_GAN = self.criterionGAN(pred_vid_fake, True) + \
        #                  self.criterionGAN(pred_speed_fake, True)
        loss_G_GAN_vid = self.criterionGAN(pred_vid_fake, True)
        loss_G_GAN_vid.backward(retain_graph=True)
        loss_G_GAN_seq = self.criterionGAN(pred_action_fake, True)
        loss_G_GAN_seq.backward(retain_graph=True)
        # Second, G(A) = B
        self.loss_G_L1_vid = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G_L1_seq = self.criterionL1(self.action_B_pred, self.action_B) * self.opt.lambda_A
        #self.loss_G_L1 = self.loss_G_L1_vid + self.loss_G_L1_seq
        self.loss_G_L1_vid.backward(retain_graph=True)
        self.loss_G_L1_seq.backward(retain_graph=True)

        self.loss_G_GAN = loss_G_GAN_vid + loss_G_GAN_seq

        self.loss_G_L1 = self.loss_G_L1_vid + self.loss_G_L1_seq

        #loss_G_vid= loss_G_GAN_vid + self.loss_G_L1_vid
        #loss_G_seq = loss_G_GAN_seq + self.loss_G_L1_seq
        #loss_G_seq.backward(retain_graph=True)
        #loss_G_vid.backward()#retain_graph=True)

        # action
        # self.action_loss = self.criterionL2(self.action,self.action_prediction)
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1  # +self.action_loss
        #self.loss_G.backward(retain_graph=True)

        # First, G(A) should fool the discriminator
        # pred_fake = self.netD(self.gen_seq)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        #
        # # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.gen_seq, self.input_seq) * 10.0  # opt.lambda_A
        #
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        #
        # self.loss_G.backward()
        # return mse loss, for print
        return self.netG.batch_mse_loss(self.input_A, self.action_B)

    def pretrain_G_step(self):
        # print(self.input_vid.size())
        g_loss = self.netG.batch_mse_loss(self.input_A, self.action_B)
        self.action_B_pred = self.netG.gen_seq
        self.optimizer_G.zero_grad()
        g_loss.backward(retain_graph=True)
        self.optimizer_G.step()
        return g_loss

    def pretrain_D_seq(self):
        label_size = list(self.action_B.size())
        label_size[2] = 1
        target_action_real = Variable(torch.ones(label_size).resize_(label_size[0], label_size[1]))
        target_action_fake = Variable(torch.zeros(label_size).resize_(label_size[0], label_size[1]))
        self.forward()

        # warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
        #               "Please ensure they have the same size.".format(self.speedX_B.size(), target_real.size()))

        d_loss = self.netD_seq.batch_bce_loss(self.action_B.cuda(), target_action_real.cuda())
        #d_loss += self.netD_vid.batch_bce_loss(self.speedX_A_pred.detach().cuda(), target_speed_fake.cuda())
        self.optimizer_D_seq.zero_grad()
        d_loss.backward()
        self.optimizer_D_seq.step()
        return d_loss

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D_vid.zero_grad()
        self.optimizer_D_seq.zero_grad()
        self.backward_D()
        self.optimizer_D_vid.step()
        self.optimizer_D_seq.step()

        self.optimizer_G.zero_grad()
        g_mse_loss = self.backward_G()
        self.optimizer_G.step()


        return g_mse_loss

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])


    def get_current_visuals(self):
        if self.dataset_mode == 'v':
            real_A = util.tensor2vid(self.real_A.data)
            fake_B = util.tensor2vid(self.fake_B.data)
            real_B = util.tensor2vid(self.real_B.data)
        else:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])



#    def get_current_visuals(self):
#       return OrderedDict([('real_A', self.input_A), ('fake_B', self.fake_B),
#                            ('real_B', self.speedX_A), ('fake_seq', self.speedX_A_pred)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_vid, 'D_vid', label, self.gpu_ids)
        self.save_network(self.netD_seq, 'D_seq', label, self.gpu_ids)
