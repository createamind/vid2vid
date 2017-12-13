 python train.py --dataroot ~/elsaW/video/ --dataset_mode v --name trial_pix2pix --model pix2pix --which_model_netG unet_128 --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --gpu_ids 1

python train.py --dataroot ~/elsaW/video/ --dataset_mode v  --model pix2pix --which_model_netG unet_256 --which_direction AtoB --norm batch --niter 10 --niter_decay 10   --batchSize 3   --name baby70-zdx  --depth 70  --max_dataset_size 15000 --continue_train

python train.py --dataroot ~/elsaW/video/ --dataset_mode v --model sensor_model --which_model_netG SensorGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 1 --name test1211 --depth 3 --max_dataset_size 15000 --output_nc 3 --input_nc 3 --sensor_types action --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids -1

python  train.py --dataroot ~/elsaW/video/ --dataset_mode v --model sensor_model --which_model_netG SensorGenerator --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --batchSize 2 --name test1211 --depth 3 --max_dataset_size 15000 --output_nc 3 --input_nc 3 --sensor_types speedX --data_dir '/data/dataset/torcs_data/**/' --input_num 2 --gpu_ids -1



