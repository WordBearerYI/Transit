#!/bin/bash

# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=v1_pose0
# path to dataset
DATA_DIR=../data/2D/v1_pose0
# training epochs
SEED=22321
LAT=16
CONV=2
EPOCH=15000
# batch size
BS=128

# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=19
# logging interval
LOG=20
MODEL=../results/2D/v1_pose0/model_best.pth
POSE=../results/2D/icp_v1_pose0/pose_est.npy
#POSE=../results/2D/v1_pose0/pose_est.npy
### training from scratch
MODE=maxpool
python train_2D.py -o $MODE --name $NAME -y $LAT -k $CONV -d $DATA_DIR -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG -s $SEED #-i $POSE -m $MODE

#### warm start
#### uncomment the following commands to run DeepMapping with a warm start. This requires an initial sensor pose that can be computed using ./script/run_icp.sh
#POSE=../results/2D/v1_pose0/pose_est.npy
#python train_2D.py --name $NAME -i $POSE -m $MODE  -d $DATA_DIR  -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG
