#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_IN_pt_fgsm_ft_resnet50
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_IN_pt_fgsm_ft_resnet50.o

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
											 -l logs/IN_pt_fgsm_ft_resnet50 \
											 -r result.json \
											 --bb_weight data/models/IN_pt_fgsm_ft_resnet50.pth \
											 --checkpoint 10 \
											 --mGPUs \
											 --num_epochs 50 