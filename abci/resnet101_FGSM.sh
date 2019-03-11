#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N frcnn_res50_FGSM
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_resnet101_FGSM.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python run_all.py -a resnet101 \
									-j 40 \
									-b 16 \
									--cuda \
									-l logs/resnet101_FGSM \
									-r result.json \
									--logger_dir logs/logger_test/logger_output\
									--bb_weight data/models/resnet101_FGSM.pth \
									--checkpoint 10 \
									--mGPUs \
									--num_epochs 50 