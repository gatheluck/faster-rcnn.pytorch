models=(resnet50 resnet101)
#train_types=(IN IN-FGSM IN-HALF IN-IN FGSM FGSM-IN FGSM-HALF FGSM-FGSM)
train_types=(IN IN-FGSM FGSM FGSM-IN)
where=mlab  #(aist mlab)

if [ ${where} = aist ]; then
	venv=anaconda3
	logdir=~/user/waseda/abci_log
	projectdir=/home/aaa10329ah/user/waseda/faster-rcnn
	gpu_ids=0,1,2,3
	env_name=faster-rcnn.pytorch
elif [ ${where} = mlab ]; then
	venv=miniconda3
	logdir=~/iccv2019/abci_log
	projectdir=~/iccv2019/faster-rcnn
	gpu_ids=0,1,2,3
	env_name=frcnn
else
	echo 'Invalid' 1>&2
  exit 1
fi

project=frcnn #name of the projects
suffix=main
mkdir -p ${project}
for model in ${models[@]}; do
	for train_type in ${train_types[@]}; do
		filename=${project}/${model}_${train_type}.sh

		name=${project}_${model}_${train_type}
		logpath=${logdir}/${name}.o

		if [ ${model} = resnet50 ]; then
			batch_size=32
		elif [ ${model} = resnet101 ]; then
			batch_size=32
		else
			echo 'Invalid' 1>&2
			exit 1
		fi

		echo -e "#!/bin/bash\n\n#$ -l rt_F=1\n#$ -l h_rt=24:00:00\n#$ -j y\n#$ -N ${name}\n#$ -o ${logpath}\n\n" > ${filename}
		echo -e "source /etc/profile.d/modules.sh\nmodule load cuda/9.0/9.0.176.4\nexport PATH=\"~/miniconda3/bin:\${PATH}\"\nsource activate ${env_name}\n" >> ${filename}

		echo -e "cd ${projectdir}" >>  ${filename}
		echo -e "python run_all.py \
			-a ${model} \
			-j 40 \
			-b ${batch_size} \
			--cuda \
			-l logs/${name} \
			-r result.json \
			--logger_dir logs/${name}/logger_out \
			--bb_weight data/models/${model}_${train_type}.pth \
			--checkpoint 10 \
			--mGPUs \
			--num_epochs 50 \
			--lr 0.001 \
			--num_max_itr 50 \
			--wd 0.0005" >> ${filename}
	done
done