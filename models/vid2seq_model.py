from collections import OrderedDict

import torch
from torch.autograd import Variable
from models.base_model import BaseModel
import models.networks as networks
import warnings

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

        # define tensors
        # 3D tensor shape (N,Cin,Din,Hin,Win)
        # self.input_vid = self.Tensor(opt.batchSize, opt.input_nc,
        #                          opt.depth, opt.fineSize, opt.fineSize)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.depth, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.depth, opt.fineSize, opt.fineSize)


        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,  # of gen filters in first conv layer
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

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
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input, is_numpy=Ture):
        """
        :param is_numpy: using numpy array or not
        :param inputs: a dict contains two inputs forms with key: 'video' and 'target_seq'
        :return:
        """
        ## numpy to torch tensor
        # input_A = input['A' if AtoB else 'B']
        # input_B = input['B' if AtoB else 'A']
        AtoB = self.opt.which_direction == 'AtoB'

        input_A = torch.from_numpy(input['A' if AtoB else 'B'])
        # print("======input A SIZE==== {0}".format(input_A.size()))
        input_B = torch.from_numpy(input['B' if AtoB else 'A'])
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.input_seq = Variable(torch.from_numpy(input["speedX"])).float()

        # convert to cuda
        if self.gpu_ids and torch.cuda.is_available():
            self.input_A = self.input_A.cuda()
            self.input_B = self.input_B.cuda()
            self.input_seq = self.input_seq.cuda()

        self.input_vid = self.input_A

        self.image_paths = input['A_paths' if AtoB else 'B_paths']


        # # numpy to torch tensor
        # if is_numpy:
        #     self.input_vid = torch.from_numpy(self.input_A)
        #     self.input_seq = torch.from_numpy(input['target_seq'])
        # else:
        #     #print(input['A'].size())
        #     self.input_vid = Variable(torch.from_numpy(input['A'])).float()
        #     #print(self.input_vid.size())
        #     self.input_seq = Variable(torch.from_numpy(input["speedX"])).float()
        # # convert to cuda
        # if self.gpu_ids and torch.cuda.is_available():
        #     self.input_vid = self.input_vid.cuda()
        #     #print(self.input_vid.size())
        #     self.input_seq = self.input_seq.cuda()

    def forward(self):
        self.gen_seq = self.netG(self.input_vid)
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)


        self.fake_B, self.speedX_pred = self.netG(self.real_A)
        # print("speedX_pred",self.speedX_pred.size())
        print("...............................")
        print(self.speedX)
        print(self.speedX_pred)






    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake seq
        if type(self.gen_seq) != Variable:
            fake_seq = Variable(self.gen_seq)
        else:
            fake_seq = self.gen_seq
        pred_fake = self.netD(fake_seq.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        label_size = list(self.input_seq.size())
        label_size[2] = 1
        pred_real = Variable(torch.ones(label_size)).cuda()
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fool the discriminator
        pred_fake = self.netD(self.gen_seq)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.gen_seq, self.input_seq) * 10.0  # opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()
        # return mse loss, for print
        return self.netG.batch_mse_loss(self.input_vid, self.input_seq)


    def pretrain_G_step(self):
        #print(self.input_vid.size())
        g_loss = self.netG.batch_mse_loss(self.input_vid, self.input_seq)
        self.gen_seq = self.netG.gen_seq
        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()
        return g_loss


    def pretrain_D_step(self):
        label_size = list(self.input_seq.size())
        label_size[2] = 1
        target_real = Variable(torch.ones(label_size).resize_(label_size[0], label_size[1] ))
        target_fake = Variable(torch.zeros(label_size).resize_(label_size[0], label_size[1] ))
        self.forward()

        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(self.input_seq.size(), target_real.size()))



        d_loss = self.netD.batch_bce_loss(self.input_seq.cuda(), target_real.cuda())
        d_loss += self.netD.batch_bce_loss(self.gen_seq.detach().cuda(), target_fake.cuda())
        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()
        return d_loss


    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

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
        return OrderedDict([('video', self.input_vid), ('fake_seq', self.gen_seq),
                            ('real_seq', self.input_seq)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
