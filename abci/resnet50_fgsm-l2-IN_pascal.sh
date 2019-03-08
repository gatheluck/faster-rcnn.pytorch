#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_res50_fgsm-l2-IN_pascal
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_resnet50_fgsm-l2-IN_pascal.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python trainval_net.py -a resnet50 \
											 -j 16 \
											 -b 32 \
											 --cuda \
											 -l logs/resnet50_fgsm-l2-IN_pascal \
											 -r result.json \
											 --bb_weight data/models/resnet50_fgsm-l2-IN.pth \
											 --checkpoint 10 \
											 --mGPUs \
											 --num_epochs 50 