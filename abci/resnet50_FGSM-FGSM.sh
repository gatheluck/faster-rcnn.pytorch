#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N res50_FGSM-FGSM
#$ -o /home/aaa10329ah/user/waseda/abci_log/frcnn_resnet50_FGSM-FGSM.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate faster-rcnn.pytorch
cd /home/aaa10329ah/user/waseda/faster-rcnn
# script

python run_all.py -a resnet50 \
									-j 40 \
									-b 32 \
									--cuda \
									-l logs/resnet50_FGSM-FGSM \
									-r result.json \
									--logger_dir logs/logger_test/logger_output\
									--bb_weight data/models/resnet50_FGSM-FGSM.pth \
									--checkpoint 10 \
									--mGPUs \
									--num_epochs 50 