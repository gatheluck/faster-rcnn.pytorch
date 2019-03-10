#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_res101_jwy-IN_bs16
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_resnet101_jwy-IN_bs16.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python run_all.py -a resnet101 \
									-j 16 \
									-b 16 \
									-lr 1e-2 \
									--step_size 8 \
									--num_epochs 10 
									--checkpoint 5 \
									--bb_weight data/models/resnet101_jwy-IN.pth \
									-l logs/resnet101_jwy-IN_bs16 \
									-r result.json \
									--cuda \
									--mGPUs \