#!/bin/bash
# path to dataset

SCENE=Home_015_1
TRAJ=traj$1
DATA_DIR=/home/mmvc/mmvc-ny-nas/Yi_Shi/data/ActiveVisionDataset/$SCENE
# trajectiory file name
# experiment name, the results will be saved to ../results/2D/${NAME}
NAME=AVD_${SCENE}_${TRAJ}
# training epochs

EPOCH=15000
# batch size
BS=16
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=35
# logging interval
GPUID=1
LOG=8
LAT=16
MODE=single
### training from scratch
python lstm_train_AVD.py -o $MODE -g $GPUID -y $LAT --name $NAME -d $DATA_DIR -t ${TRAJ}.txt -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG
~                                            
