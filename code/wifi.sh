#!/bin/bash

tar_id=3
useMB=True

if [ $tar_id == 1 ]
then
	dag_mat='t2t3_dag.npz'
elif [ $tar_id == 2 ]
then
	dag_mat='t1t3_dag.npz'
else
	dag_mat='t1t2_dag.npz'
fi

if [ $useMB == True ]
then
    if [ $tar_id == 1 ]
    then
        dim_z=17
        i_dim=17
    else
        dim_z=16
        i_dim=16
    fi
else
    dim_z=67
    i_dim=67
fi
echo $dag_mat
echo $i_dim
echo $dim_z

## baseline, mlp classifier
#for seed in $(seq 1 10)
#do
#if [ $useMB == True ]
#then
#	python train_poolnn.py --dataset DatasetWifi --cuda --num_workers 2 --tar_id $tar_id \
#	--seed $seed --num_class 19 --num_domain 3 --num_train 700 --idim $i_dim --trainer DA_Poolnn \
#	--estimate ML --AC_weight 1.0 --TAR_weight 1.0 --G_model MLP_Generator --D_model MLP_Classifier \
#	--D_mlp_layers 1 --D_mlp_nodes 32 --dim_z $dim_z --dim_y 19 --dim_d 1 \
#	--num_epochs 1000 --batch_size 210 --dag_mat_file $dag_mat
#else
#	python train_poolnn.py --dataset DatasetWifi --cuda --num_workers 2 --tar_id $tar_id \
#	--seed $seed --num_class 19 --num_domain 3 --num_train 700 --idim $i_dim --trainer DA_Poolnn \
#	--estimate ML --AC_weight 1.0 --TAR_weight 1.0 --G_model MLP_Generator --D_model MLP_Classifier \
#	--D_mlp_layers 1 --D_mlp_nodes 32 --dim_z $dim_z --dim_y 19 --dim_d 1 --useMB \
#	--num_epochs 1000 --batch_size 210 --dag_mat_file $dag_mat
#fi
#done

dim_z=1
for seed in $(seq 1 10)
do
if [ $useMB == True ]
then
	python train_mmd.py --dataset DatasetWifi --cuda --num_workers 2 --tar_id $tar_id \
	--seed $seed --num_class 19 --num_domain 3 --num_train 700 --idim $i_dim --trainer DA_Infer_JMMD_DAG \
	--estimate Bayesian --AC_weight 1.0 --TAR_weight 1.0 --G_model DAG_Generator --D_model MLP_Classifier \
	--G_mlp_nodes 32 --D_mlp_nodes 64 --dim_z $dim_z --dim_y 1 --dim_d 1 \
	--num_epochs 500 --batch_size 210 --warmup 200 --dag_mat_file $dag_mat --SRC_weight 0.0 --train_mode m0
else
	python train_mmd.py --dataset DatasetWifi --cuda --num_workers 2 --tar_id $tar_id \
	--seed $seed --num_class 19 --num_domain 3 --num_train 700 --idim $i_dim --trainer DA_Infer_JMMD_DAG \
	--estimate Bayesian --AC_weight 1.0 --TAR_weight 1.0 --G_model DAG_Generator --D_model MLP_Classifier \
	--G_mlp_nodes 32 --D_mlp_nodes 64 --dim_z $dim_z --dim_y 1 --dim_d 1 --useMB \
	--num_epochs 500 --batch_size 210 --warmup 200 --dag_mat_file $dag_mat --SRC_weight 0.0 --train_mode m0
fi
done

