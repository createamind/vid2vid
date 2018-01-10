 python train.py --dataroot ~/elsaW/video/ --dataset_mode v --name trial_pix2pix --model pix2pix --which_model_netG unet_128 --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --gpu_ids 1

python train.py --dataroot ~/elsaW/video/ --dataset_mode v  --model pix2pix --which_model_netG unet_256 --which_direction AtoB --norm batch --niter 10 --niter_decay 10   --batchSize 3   --name baby70-zdx  --depth 70  --max_dataset_size 15000 --continue_train

python train.py --dataroot ~/elsaW/video/ --dataset_mode v --model sensor_model --which_model_netG SensorGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name test1211 --depth 3 --max_dataset_size 15000 --output_nc 3 --input_nc 3 --sensor_types action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids -1

python  train.py --dataroot ~/elsaW/video/ --dataset_mode v --model sensor_model --which_model_netG SensorGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 2 --name test1211 --depth 3 --max_dataset_size 15000 --output_nc 3 --input_nc 3 --sensor_types speedX --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids -1



python  train.py --dataroot ~/elsaW/video/ --dataset_mode v --model sensor_model --which_model_netG SensorGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name speedx --depth 18 --max_dataset_size 15000 --output_nc 3 --input_nc 3 --sensor_types speedX --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 2  --display_freq 50 --continue_train


python train.py --dataset_mode v --model vid2seq_model --which_model_netG SequenceGenerator --which_model_netD SequenceDiscriminator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 4 --name speedxtest111111 --depth 5 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 0


python train.py --dataset_mode v --model vid2seq_model --which_model_netG SequenceGenerator --which_model_netD SequenceDiscriminator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name speedxtest111111 --depth 10 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 2

# speed_pred

python train.py --dataset_mode v --model vid2seq_model --which_model_netG SequenceGenerator --which_model_netD_seq SequenceDiscriminator --which_model_netD_vid basic --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name speedx-video-test-1 --depth 20 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 0


#angle pred
python train.py --dataset_mode v --model vid2seq_model --which_model_netG SequenceGenerator --which_model_netD_seq SequenceDiscriminator --which_model_netD_vid basic --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name angle-video-test-1-predB --depth 30 --max_dataset_size 10000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 1 --print_freq 3 --display_freq 3



python train.py --dataset_mode v --model vid2seq_model --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name
modular-test-1 --depth 30 --max_dataset_size 10000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids
 0 --print_freq 10


python train.py --dataset_mode v --model vid2seq_model --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name adversarial-speedB-test-1 --depth 30 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 0 --print_freq 10

python train.py --dataset_mode v --model vid2seq_model --which_model_netG_seq SeqRNNGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name adversarial-speedB-rnn-test-1 --depth 30 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 1 --print_freq 50 --train_mode seq_only

python train.py --dataset_mode v --model vid2seq_model --which_model_netG_seq SeqRNNGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name adversarial-speedB-rnn-test-1 --depth 30 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 1 --print_freq 50 --train_mode seq_only  --continue_train

python train.py --dataset_mode v --model vid2seq_model --which_model_netG_seq SeqRNNGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name temptest --depth 30 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids 1 --print_freq 1 --train_mode seq_only --display_freq 1 --pretrain True


CUDA_VISIBLE_DEVICES=0
sudo nvidia-smi -i 1 -pl 350

#one env

python train.py --dataset_mode v --model vid2seq_model --which_model_netG_seq SeqRNNGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name supervised-speedA-rnn-oneenv-test-1 --depth 30 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir /data/dataset/torcs-onebigenv-data/aalborg/ --input_num 2 --gpu_ids 2 --train_mode seq_only --print_freq 21 --pretrain

###18.1.10
 2049  mv  2[0-9][0-9][0-9][0-9].*  ../aalborg2.2/
  2050  mv  3[0-9][0-9][0-9][0-9].*  ../aalborg2.3/
   2051  mv  1[0-9][0-9][0-9][0-9].*  ../aalborg2.1/
python train.py --dataset_mode v --model vid2seq_model --which_model_netG_seq SeqRNNGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name supervised-speedA-rnn-oneenv-moredirsplit-test-1 --depth 30 --max_dataset_size 5000 --output_nc 3 --input_nc 3 --sensor_types angle,speedX,action --data_dir /data/dataset/torcs-onebigenv-data/**/ --input_num 2 --gpu_ids 2 --train_mode seq_only --print_freq 21 --pretrain




