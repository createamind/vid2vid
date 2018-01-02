import time, os, cv2
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import ntpath
import numpy as np
import skvideo.io
import time

output_video = False
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
print(dataset)
dataset_size = len(data_loader)
print('#training videos = %d' % dataset_size)

model = create_model(opt)
opt.results_dir = './results/'
total_steps = 0
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))


def ck_array(i, o):
    i_A = np.transpose(127.5 * (i['A'] + 1.)[0], (1, 2, 3, 0))
    i_B = np.transpose(127.5 * (i['B'] + 1.)[0], (1, 2, 3, 0))
    o_A = o['real_A']
    o_B = o['real_B']
    a_diff = i_A - o_A
    b_diff = i_B - o_B
    print('diffa', a_diff.max(), a_diff.min(), a_diff.mean())
    print('diffb', b_diff.max(), b_diff.min(), b_diff.mean())


def save_videos(web_dir, visuals, vid_path, epoch):
    vid_dir = os.path.join(web_dir, 'videos')
    # name = ntpath.basename(vid_path).split('.')[0]
    # add data generation time as name
    name = time.strftime('%Y%m%d-%H%M%S')
    # print("vid_dir: {}".format(vid_dir))
    # print("name: {}".format(name))

    A = visuals['real_A'][:, :, :, :3]
    # print("="*20 + str(A.shape))
    last_A = np.tile(A[-1], (A.shape[0], 1, 1, 1))
    # print("A_last shape: {}".format(A[-1].shape))
    # print('last_A: {}'.format(last_A.shape))
    B = visuals['real_B'][:, :, :, :3]
    first_B = np.tile(B[0], (A.shape[0], 1, 1, 1))
    fake = visuals['fake_B'][:, :, :, :3]
    first_fake = np.tile(fake[0], (A.shape[0], 1, 1, 1))
    black = np.ones_like(A)
    last_fake = np.tile(fake[-1], (A.shape[0], 1, 1, 1))
    blackforA = np.concatenate((first_B, first_fake), axis=1)
    blackforBC = np.concatenate((last_A, last_fake), axis=1)

    vid_A = np.concatenate((A, fake), axis=1)
    vid_A2 = np.concatenate((vid_A, blackforA), axis=2)
    vid_BC = np.concatenate((B, fake), axis=1)
    vid_BC2 = np.concatenate((blackforBC, vid_BC), axis=2)
    vid_numpy = np.concatenate((vid_A2, vid_BC2), axis=0)
    # print("output_img_shape: {}".format(vid_numpy.shape))

    # vid_numpy = np.concatenate((visuals['real_A'], visuals['real_B'], visuals['fake_B']), axis=2)
    # print(vid_numpy.shape)
    save_path = os.path.join(vid_dir, str(epoch)) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = name + '_' + '.mp4'

    skvideo.io.vwrite(save_path + save_name, vid_numpy,
                      inputdict={'-r': '12'},
                      outputdict={'-r': '12'})
    print('save video at ', save_path + save_name)

    if opt.output_nc > 3:
        # Depth Video
        print("visuals", visuals['real_A'].shape)
        dA = visuals['real_B'][:, :, :, 3]
        dB = visuals['real_B'][:, :, :, 4]
        dfake = visuals['fake_B'][:, :, :, 4]
        dfake_ = visuals['fake_B'][:, :, :, 3]

        print("=" * 20 + str(dA.shape))
        dlast_A = np.tile(dA[-1], (dA.shape[0], 1, 1))  # last frame
        dlast_A_ = np.tile(dfake_[-1], (dA.shape[0], 1, 1))  # last frame
        # print("A_last shape: {}".format(A[-1].shape))
        # print('last_A: {}'.format(last_A.shape))

        dfirst_B = np.tile(dB[0], (dA.shape[0], 1, 1))

        dfirst_fake = np.tile(dfake[0], (dA.shape[0], 1, 1))
        dblack = np.ones_like(dlast_A)
        dblackforA = np.concatenate((dfirst_B, dfirst_fake), axis=1)  # first frame
        print("=" * 20 + "dlastA" + str(dlast_A.shape))
        print("=" * 20 + "dblack" + str(dblack.shape))
        dblackforBC = np.concatenate((dlast_A, dlast_A_), axis=1)  ##replace dblack with dlast_A_

        dvid_A = np.concatenate((dA, dfake_), axis=1)  # replace dblack with dfake_
        dvid_A2 = np.concatenate((dvid_A, dblackforA), axis=2)  # time within A
        dvid_BC = np.concatenate((dB, dfake), axis=1)
        dvid_BC2 = np.concatenate((dblackforBC, dvid_BC), axis=2)
        dvid_numpy = np.concatenate((dvid_A2, dvid_BC2), axis=0)
        # print("output_img_shape: {}".format(vid_numpy.shape))

        # vid_numpy = np.concatenate((visuals['real_A'], visuals['real_B'], visuals['fake_B']), axis=2)
        # print(vid_numpy.shape)

        dsave_name = name + '_depth' + '.mp4'

        skvideo.io.vwrite(save_path + dsave_name, dvid_numpy,
                          inputdict={'-r': '12'},
                          outputdict={'-r': '12'})
        print('save depth video at ', save_path + dsave_name)

    while output_video:

        for i in range(vid_numpy.shape[0]):
            save_name = name + '_' + str(i) + '.png'
            # save_name =  str(i) + '.png'

            img = vid_numpy[i][:, :, ::-1]
            # print(img.shape)
            print('save path ', save_path + save_name)
            cv2.imwrite(save_path + save_name, img)


# pretrain generator

# print(range(opt.epoch_count, opt.niter + opt.niter_decay + 1))
#
# print('=' * 20 + 'Pre-train Generator' + '=' * 20)
# for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
#     # for epoch in range(1):
#     epoch_start_time = time.time()
#     epoch_iter = 0
#
#     for i, data in enumerate(dataset):
#
#         # for key, value in data.items():
#         #     print(value.shape)
#
#         iter_start_time = time.time()
#         # visualizer.reset()
#         total_steps += opt.batchSize
#         epoch_iter += opt.batchSize
#         model.set_input(data)
#         g_loss = model.pretrain_G_step()
#         if total_steps % opt.print_freq == 0:
#             print("epoch: {}, iter: {}, loss: {}, time: {} seconds/batch".format(epoch,
#                                                                                  i, g_loss.data[0], (
#                                                                                          time.time() - iter_start_time) / opt.batchSize))
#             print("target seq:\n {} \ngenerated seq: {}".format(model.input_seq, model.gen_seq))
#         if total_steps % opt.save_latest_freq == 0:
#             print('saving the latest model (epoch %d, total_steps %d)' %
#                   (epoch, total_steps))
#             model.save('latest')
#
# total_steps = 0
#
# print(range(opt.epoch_count, opt.niter + opt.niter_decay + 1))
#
# # pre-train discriminator
# print('=' * 20 + 'Pre-train Discriminator' + '=' * 20)
# for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
#     epoch_start_time = time.time()
#     epoch_iter = 0
#
#     for i, data in enumerate(dataset):
#         iter_start_time = time.time()
#         # visualizer.reset()
#         total_steps += opt.batchSize
#         epoch_iter += opt.batchSize
#         model.set_input(data)
#         d_loss = model.pretrain_D_step()
#         if total_steps % opt.print_freq == 0:
#             print("epoch: {}, iter: {}, loss: {}, time: {} seconds/batch".format(epoch,
#                                                                                  i, d_loss.data[0], (
#                                                                                          time.time() - iter_start_time) / opt.batchSize))
#         if total_steps % opt.save_latest_freq == 0:
#             print('saving the latest model (epoch %d, total_steps %d)' %
#                   (epoch, total_steps))
#             model.save('latest')

# total_steps = 0
# adversarial training
print('=' * 20 + 'Adversarial Training' + '=' * 20)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        if 0:
            AB_path = "./data/data/"
            trial_A = np.ones((opt.batchSize, opt.input_nc, opt.depth, opt.fineSize, opt.fineSize))
            trial_B = np.ones((opt.batchSize, opt.output_nc, opt.depth, opt.fineSize, opt.fineSize))
            data = {'A': trial_A, 'B': trial_B, 'A_paths': AB_path, 'B_paths': AB_path}

        # data = dict(data)

        #print(data)
        # print(data['speedX'].shape)

        iter_start_time = time.time()
        # visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        g_mse_loss = model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0

            visuals = model.get_current_visuals()
            #ck_array(data, visuals)
            vid_path = model.get_image_paths()

            # print(visuals)
            print('process video... %s,progress %d' % (vid_path, i) )
            save_videos(web_dir, visuals, vid_path, epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print("epoch: {}, iter: {}, g-mse-loss: {}, time: {} seconds/batch".format(
                epoch, i, g_mse_loss.data[0], (time.time() - iter_start_time) / opt.batchSize))
            print(model.get_current_errors())
            print("seq A :\n {} target seq:\n {} \ngenerated seq: {}".format(model.speedX_A,model.speedX_B, model.speedX_B_pred))
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # if opt.display_id > 0:
            # visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        if total_steps % 20010 == 0:
            print('saving the 20010 model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save(total_steps)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
