import models.networks as networks
import torch
from models.vid2seq_model import Vid2SeqModel
from torch.optim import lr_scheduler

generator = networks.SequenceGenerator(input_nc=3, output_nc=1, rnn_input_sie=232)
# batch_size x channels x depth x height x width
fake_vid = torch.autograd.Variable(torch.randn(1, 3, 30, 32, 32))
fake_seq = torch.autograd.Variable(torch.randn(1, 30, 2))
print('encoded vid: ', generator.video_encoder(fake_vid).size())
g_out = generator.forward(fake_vid)
print('g_out: ', g_out.size())
g_target = torch.autograd.Variable(torch.randn(g_out.shape))
g_loss = generator.batch_mse_loss(fake_vid, g_target)
print('g_loss: ', g_loss)
discriminator = networks.SequenceDiscriminator(g_out.size()[2], 100)
d_out = discriminator.forward(g_out)
print('d_out: ', d_out.size())
pred = discriminator.batch_classify(g_out)
print('d_pred: ', pred.size())
target = torch.autograd.Variable(torch.zeros(pred.size()))
print('target: ', target.size())
d_loss = discriminator.batch_bce_loss(g_out, target)
print('d_loss: ', d_loss)

print('=' * 10, "Test GAN Model", '=' * 10)
model = Vid2SeqModel()
model.netG = generator
model.netD = discriminator
model.isTrain = True
model.gpu_ids = None
model.criterionGAN = networks.GANLoss(use_lsgan=True)
model.criterionL1 = torch.nn.L1Loss()

# initialize optimizers
model.schedulers = []
model.optimizers = []
model.optimizer_G = torch.optim.Adam(model.netG.parameters(),
                                     lr=0.001, betas=(0.5, 0.999))
model.optimizer_D = torch.optim.Adam(model.netD.parameters(),
                                     lr=0.001, betas=(0.5, 0.999))
model.optimizers.append(model.optimizer_G)
model.optimizers.append(model.optimizer_D)
for optimizer in model.optimizers:
    model.schedulers.append(lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1))

epoch = 10
for i in range(epoch):
    print('epoch: ', i)
    _inputs = {'video': fake_vid, 'target_seq': fake_seq}
    print('input vid: {}, input seq: {}'.format(_inputs['video'].size(), _inputs['target_seq'].size()))
    model.set_input(_inputs)
    model.forward()
    print('generated sequence: ', model.gen_seq.size())
    model.backward_D()
    model.backward_G()
    print('current error: ', str(model.get_current_errors()))
