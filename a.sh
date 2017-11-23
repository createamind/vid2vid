 python train.py --dataroot ~/elsaW/video/ --dataset_mode v --name trial_pix2pix --model pix2pix --which_model_netG unet_128 --which_direction AtoB --norm batch --niter 10 --niter_decay 10 --gpu_ids 1

python train.py --dataroot ~/elsaW/video/ --dataset_mode v  --model pix2pix --which_model_netG unet_256 --which_direction AtoB --norm batch --niter 10 --niter_decay 10   --batchSize 3   --name baby70-zdx  --depth 70  --max_dataset_size 15000 --continue_train






