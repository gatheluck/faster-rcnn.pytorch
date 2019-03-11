#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_res101_pth-IN_bs04
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_resnet101_pth-IN_bs04.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python run_all.py -a resnet101 \
									-j 16 \
									-b 4 \
									--lr 0.01 \
									--step_size 8 \
									--num_epochs 10 \
									--checkpoint 5 \
									--bb_weight ./data/models/resnet101_pth-IN.pth \
									-l ./logs/resnet101_pth-IN_bs04 \
									-r result.json \
									--cuda \
									--mGPUs