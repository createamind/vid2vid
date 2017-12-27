import models.networks as networks
import torch
from models.vid2seq_model import Vid2SeqModel
from torch.optim import lr_scheduler
from torch.autograd import Variable

generator = networks.SequenceGenerator(input_nc=3, output_nc=1, rnn_input_sie=232)
# batch_size x channels x depth x height x width
fake_vid = torch.autograd.Variable(torch.randn(1, 3, 30, 32, 32))
fake_seq = torch.autograd.Variable(torch.randn(1, 30, 2))
print('encoded vid: ', generator.video_encoder(fake_vid).size())
g_out = generator.forward(fake_vid)
print('g_out: ', g_out.size())
g_target = torch.autograd.Variable(torch.randn(g_out.size()))
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

_inputs = {'video': fake_vid, 'target_seq': fake_seq}

# pre-train generator
print('='*20 + 'Pre-train Generator' + '='*20)
for i in range(epoch):
    print('pre-train generator, epoch: ', i)
    print('input vid: {}, input seq: {}'.format(_inputs['video'].size(), _inputs['target_seq'].size()))
    model.set_input(_inputs)
    g_loss = model.netG.batch_mse_loss(model.input_vid, model.input_seq)
    model.optimizer_G.zero_grad()
    g_loss.backward()
    model.optimizer_G.step()
    print('current error: ', g_loss.data[0])

# pre-train discriminator TODO should not train D like this, just for test sake
print('='*20 + 'Pre-train Discriminator' + '='*20)
for i in range(epoch + 5):  # stronger discriminator
    print('pre-train discriminator, epoch: ', i)
    print('input vid: {}, input seq: {}'.format(_inputs['video'].size(), _inputs['target_seq'].size()))
    model.set_input(_inputs)
    label_size = list(model.input_seq.size())
    label_size[2] = 1
    target_real = Variable(torch.ones(label_size).resize_(label_size[0], label_size[1]))
    target_fake = Variable(torch.zeros(label_size).resize_(label_size[0], label_size[1]))
    model.forward()
    d_loss = model.netD.batch_bce_loss(model.input_seq, target_real)
    d_loss += model.netD.batch_bce_loss(model.gen_seq.detach(), target_fake)
    model.optimizer_D.zero_grad()
    d_loss.backward()
    model.optimizer_D.step()
    print('current error: ', d_loss.data[0])

# adversarial training
print('='*20 + 'Adversarial Training' + '='*20)
for i in range(epoch):
    print('adversarial training, epoch: ', i)
    print('input vid: {}, input seq: {}'.format(_inputs['video'].size(), _inputs['target_seq'].size()))
    model.set_input(_inputs)
    model.optimize_parameters()
    print('current error: ', str(model.get_current_errors()))
