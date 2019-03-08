#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_res101_pth-IN_pascal
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_resnet101_pth-IN_pascal.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python trainval_net.py -a resnet101 \
											 -j 16 -b 16 \
											 --cuda \
											 -l logs/resnet101_pth-IN_pascal \
											 -r result.json \
											 --bb_weight data/models/resnet101_pth-IN.pth \
											 --checkpoint 10 \
											 --mGPUs \
											 --num_epochs 50 
