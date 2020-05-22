#!/usr/bin/env bash
for seed in $(seq 2 27)
do
    python train_multi_adv.py --num_class=2 --num_domain=6 --num_train=500 --source_dataset=DatasetFlow5 \
    --target_dataset=DatasetFlow5 --G_model=DAGAN_MLP_Gen --D_model=DAGAN_MLP_Dis --idim_a=4 \
    --num_layer_mlp=1 --num_nodes_mlp=64 --dim_z=4 --dim_y=2 --dim_d=1 --dag_mat_file=None --seed=$seed \
    --batch_size=50 --num_epochs=500 --trainer=DAGAN_Multi_Adv
done