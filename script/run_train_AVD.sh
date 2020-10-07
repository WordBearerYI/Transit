#!/bin/bash

NUM=$1
SCENE=Home_015_1
# path tNUM=$1
TRAJ=traj${NUM}
DATA_DIR=/home/mmvc/mmvc-ny-nas/Yi_Shi/data/ActiveVisionDataset/$SCENE
# trajectiory file name
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=Z_AVD_${SCENE}_${TRAJ}
# training epochs

# training epochs
EPOCH=15000
# batch size
BS=16
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=35
# logging interval
LOG=2
LAT=16
### training from scratch
python train_AVD.py --scene_index ${NUM} -y $LAT --name $NAME -d $DATA_DIR -t ${TRAJ}.txt -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG
