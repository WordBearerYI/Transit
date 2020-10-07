#!/bin/bash

NUM=$1
CHECKPOINT_DIR="../results/AVD/Z_AVD_Home_015_1_traj${NUM}/"
python eval_vis_AVD.py -c $CHECKPOINT_DIR
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR
