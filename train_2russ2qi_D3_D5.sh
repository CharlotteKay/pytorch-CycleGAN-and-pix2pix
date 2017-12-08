# training. Tune number of layers in D
python train.py --dataroot datasets/2russ2qi --display_id 1 --name 2russ2qi_D3_D5 --model cycle_gan --checkpoints_dir checkpoints/ --gpu_ids 3 --save_epoch_freq 20 --n_layers_D1 3 --n_layers_D2 5 --display_port 8097

# testing
#python test.py --dataroot datasets/2russ2qi --name 2russ2qi_D5 --model cycle_gan --phase test
