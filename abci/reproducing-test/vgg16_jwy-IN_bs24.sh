#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_vgg16_jwy-IN_bs24
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_vgg16_jwy-IN_bs24.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python run_all.py -a vgg16 \
									-j 16 \
									-b 24 \
									-lr 1e-2 \
									--step_size 10 \
									--num_epochs 10 
									--checkpoint 5 \
									--bb_weight data/models/vgg16_jwy-IN.pth \
									-l logs/vgg16_jwy-IN_bs24 \
									-r result.json \
									--cuda \
									--mGPUs 