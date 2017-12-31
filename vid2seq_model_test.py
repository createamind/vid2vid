import models.networks as networks
import torch
from models.vid2seq_model import Vid2SeqModel
from torch.optim import lr_scheduler
from torch.autograd import Variable
from options.train_options import TrainOptions

generator = networks.SequenceGenerator(input_nc=3, output_nc=3, rnn_input_size=3072)
# batch_size x channels x depth x height x width
input_vid = torch.autograd.Variable(torch.randn(1, 3, 10, 32, 32))
input_seq = torch.autograd.Variable(torch.randn(1, 20, 1))
print('encoded vid: ', generator.video_encoder(input_vid).size())
generator.forward(input_vid)
print('g_out: ', generator.gen_seq.size())
g_target = torch.autograd.Variable(torch.randn(generator.gen_seq.size()))
g_loss = generator.batch_mse_loss(input_vid, g_target)
print('g_loss: ', g_loss)

# conv_f = torch.nn.Conv3d(3, 3, kernel_size=(3, 8, 8), padding=(1, 3, 3))
# print(conv_f(input_vid))

discriminator = networks.SequenceDiscriminator(generator.gen_seq.size()[2] * 2, 100)    # concat seq
d_in = torch.cat([generator.gen_seq, generator.gen_seq], 2)
d_out = discriminator.forward(d_in)
print('d_out: ', d_out.size())
pred = discriminator.batch_classify(d_in)
print('d_pred: ', pred.size())
target = torch.autograd.Variable(torch.zeros(pred.size()))
print('target: ', target.size())
d_loss = discriminator.batch_bce_loss(d_in, target)
print('d_loss: ', d_loss)

print('=' * 10, "Test GAN Model", '=' * 10)
model = Vid2SeqModel()
model.netG = generator
model.netD_seq = discriminator
model.netD_vid = networks.NLayerDiscriminator(input_nc=6)
model.isTrain = True
model.gpu_ids = None

model.opt = TrainOptions()
model.opt.which_direction = 'AtoB'
model.opt.lambda_A = 10.0
model.criterionGAN = networks.GANLoss(use_lsgan=True)
model.criterionL1 = torch.nn.L1Loss()

# initialize optimizers
model.schedulers = []
model.optimizers = []
model.optimizer_G = torch.optim.Adam(model.netG.parameters(),
                                     lr=0.001, betas=(0.5, 0.999))
model.optimizer_D_seq = torch.optim.Adam(model.netD_seq.parameters(),
                                     lr=0.001, betas=(0.5, 0.999))
model.optimizer_D_vid = torch.optim.Adam(model.netD_seq.parameters(),
                                     lr=0.001, betas=(0.5, 0.999))

model.optimizers.append(model.optimizer_G)
model.optimizers.append(model.optimizer_D_seq)
model.optimizers.append(model.optimizer_D_vid)
for optimizer in model.optimizers:
    model.schedulers.append(lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1))


epoch = 1
_inputs = {'A': input_vid.data.numpy(), 'B': input_vid.data.numpy(), 'speedX': input_seq.data.numpy()}

# pre-train generator
print('='*20 + 'Pre-train Generator' + '='*20)
for i in range(epoch):
    print('pre-train generator, epoch: ', i)
    # print('inputs: {}'.format(_inputs))
    model.set_input(_inputs)
    g_loss = model.netG.batch_mse_loss(model.input_A, model.speedX)
    model.optimizer_G.zero_grad()
    g_loss.backward()
    model.optimizer_G.step()
    print('current error: ', g_loss.data[0])

# pre-train discriminator TODO should not train D like this, just for test sake
print('='*20 + 'Pre-train Discriminator' + '='*20)
for i in range(epoch):  # stronger discriminator
    print('pre-train discriminator, epoch: ', i)
    # print('input vid: {}, input seq: {}'.format(_inputs['A'].size(), _inputs['B'].size()))
    model.set_input(_inputs)
    label_size = list(model.speedX.size())
    label_size[2] = 1
    target_real = Variable(torch.ones(label_size).resize_(label_size[0], label_size[1]))
    target_fake = Variable(torch.zeros(label_size).resize_(label_size[0], label_size[1]))
    model.forward()
    fake = torch.cat([model.speedX, model.speedX_pred], 2)
    real = torch.cat([model.speedX, model.speedX], 2)
    d_loss = model.netD_seq.batch_bce_loss(real, target_real)
    d_loss += model.netD_seq.batch_bce_loss(fake, target_fake)
    model.optimizer_D_seq.zero_grad()
    d_loss.backward()
    model.optimizer_D_seq.step()
    print('current error: ', d_loss.data[0])

# adversarial training
print('='*20 + 'Adversarial Training' + '='*20)
for i in range(epoch):
    print('adversarial training, epoch: ', i)
    # print('input vid: {}, input seq: {}'.format(_inputs['A'].size(), _inputs['speedX'].size()))
    model.set_input(_inputs)
    model.optimize_parameters()
    print('current error: ', str(model.get_current_errors()))
