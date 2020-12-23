#!/bin/bash
export HDF5_USE_FILE_LOCKING='FALSE'
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--loss_type Twin_AC --AC \
--AC_weight 1.0 \
--G_shared \
--n_domain 4 \
--shuffle --batch_size 200 \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000 \
--num_D_steps 4 --num_G_steps 1 --G_lr 2e-4 --D_lr 2e-4 \
--source_dataset mnist,mnist_m,svhn,syn_digits --target_dataset mnist --num_workers 16 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 --G_ch 64 --D_ch 64 \
--G_init N02 --D_init N02 \
--test_every 8000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 2019 \
--ema  --use_ema --ema_start 10000 \
