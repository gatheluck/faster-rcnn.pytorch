from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
torch.backends.cudnn.benchmark = True
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from option import TrainOptions
from model.faster_rcnn.ours_resnet import resnet 

class sampler(Sampler):
	def __init__(self, train_size, batch_size):
		self.num_data = train_size
		self.num_per_batch = int(train_size / batch_size)
		self.batch_size = batch_size
		self.range = torch.arange(0,batch_size).view(1, batch_size).long()
		self.leftover_flag = False
		if train_size % batch_size:
			self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
			self.leftover_flag = True

	def __iter__(self):
		rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
		self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

		self.rand_num_view = self.rand_num.view(-1)

		if self.leftover_flag:
			self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

		return iter(self.rand_num_view)

	def __len__(self):
		return self.num_data


def main():
	opt = TrainOptions().parse()

	if opt.dataset_rcnn == "pascal_voc":
		opt.imdb_name = "voc_2007_trainval"
		opt.imdbval_name = "voc_2007_test"
		opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif opt.dataset_rcnn == "pascal_voc_0712":
		opt.imdb_name = "voc_2007_trainval+voc_2012_trainval"
		opt.imdbval_name = "voc_2007_test"
		opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif opt.dataset_rcnn == "coco":
		opt.imdb_name = "coco_2014_train+coco_2014_valminusminival"
		opt.imdbval_name = "coco_2014_minival"
		opt.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	elif opt.dataset_rcnn == "imagenet":
		opt.imdb_name = "imagenet_train"
		opt.imdbval_name = "imagenet_val"
		opt.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	elif opt.dataset_rcnn == "vg":
		opt.imdb_name = "vg_150-50-50_minitrain"
		opt.imdbval_name = "vg_150-50-50_minival"
		opt.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

	# config file
	opt.cfg_file = "cfgs/{}_ls.yml".format(opt.model) if opt.large_scale else "cfgs/{}.yml".format(opt.model)

	if opt.cfg_file is not None:
		cfg_from_file(opt.cfg_file)
	if opt.set_cfgs is not None:
		cfg_from_list(opt.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	# train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = opt.use_gpu
	imdb, roidb, ratio_list, ratio_index = combined_roidb(opt.imdb_name)
	train_size = len(roidb)

	print('{:d} roidb entries'.format(len(roidb)))

	# output dir
	output_dir = opt.log_dir + "/" + opt.model + "/" + opt.dataset_rcnn
	if not os.path.exists(output_dir):	os.makedirs(output_dir)

	sampler_batch = sampler(train_size, opt.batch_size)
	dataset_rcnn = roibatchLoader(roidb, ratio_list, ratio_index, opt.batch_size, imdb.num_classes, training=True)
	dataloader = torch.utils.data.DataLoader(dataset_rcnn, batch_size=opt.batch_size, sampler=sampler_batch, num_workers=opt.num_workers)

	# initilize the tensor holder.
	im_data = torch.FloatTensor(1).to(opt.device)
	im_info = torch.FloatTensor(1).to(opt.device)
	num_boxes = torch.LongTensor(1).to(opt.device)
	gt_boxes = torch.FloatTensor(1).to(opt.device)

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if opt.use_gpu:	cfg.CUDA = True

	print("imdb.classes: ", imdb.classes)

	# initilize the network here.
	if opt.model == 'lenet':
		raise NotImplementedError
	elif opt.model == 'alexnet':
		raise NotImplementedError
	elif opt.model == 'resnet18':
		fasterRCNN = resnet(imdb.classes, 18, opt, class_agnostic=opt.class_agnostic)
	elif opt.model == 'resnet34':
		fasterRCNN = resnet(imdb.classes, 34, opt, class_agnostic=opt.class_agnostic)
	elif opt.model == 'resnet50':
		fasterRCNN = resnet(imdb.classes, 50, opt, class_agnostic=opt.class_agnostic)
	elif opt.model == 'resnet101':
		fasterRCNN = resnet(imdb.classes, 101, opt, class_agnostic=opt.class_agnostic)
	else:
		raise NotImplementedError
		pdb.set_trace()

	fasterRCNN.create_architecture()

	lr = cfg.TRAIN.LEARNING_RATE
	lr = opt.lr

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), 'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
			else:
				params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

	if opt.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params)
	elif opt.optimizer == "sgd":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	if opt.resume:
		load_name = os.path.join(output_dir,'faster_rcnn_{}_{}_{}.pth'.format(opt.checksession, opt.checkepoch, opt.checkpoint))
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		opt.session = checkpoint['session']
		opt.start_epoch = checkpoint['epoch']
		fasterRCNN.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		lr = optimizer.param_groups[0]['lr']
		if 'pooling_mode' in checkpoint.keys():
			cfg.POOLING_MODE = checkpoint['pooling_mode']
		print("loaded checkpoint %s" % (load_name))

	if opt.mGPUs:	fasterRCNN = nn.DataParallel(fasterRCNN)
	fasterRCNN = fasterRCNN.to(opt.device)

	iters_per_epoch = int(train_size / opt.batch_size)

	if opt.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	# training
	for epoch in range(opt.start_epoch, opt.max_epochs + 1):
		# setting to train mode
		fasterRCNN.train()
		loss_temp = 0
		start = time.time()

		if epoch % (opt.lr_decay_step + 1) == 0:
			adjust_learning_rate(optimizer, opt.lr_decay_gamma)
			lr *= opt.lr_decay_gamma

		data_iter = iter(dataloader)

		for step in range(iters_per_epoch):

			data = next(data_iter)
			im_data.data.resize_(data[0].size()).copy_(data[0])
			im_info.data.resize_(data[1].size()).copy_(data[1])
			gt_boxes.data.resize_(data[2].size()).copy_(data[2])
			num_boxes.data.resize_(data[3].size()).copy_(data[3])

			fasterRCNN.zero_grad()
			rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

			loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
			loss_temp += loss.item()

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if step % opt.disp_interval == 0:
				end = time.time()
				if step > 0: 
					loss_temp /= (opt.disp_interval + 1)

				if opt.mGPUs:
					loss_rpn_cls = rpn_loss_cls.mean().item()
					loss_rpn_box = rpn_loss_box.mean().item()
					loss_rcnn_cls = RCNN_loss_cls.mean().item()
					loss_rcnn_box = RCNN_loss_bbox.mean().item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt
				else:
					loss_rpn_cls = rpn_loss_cls.item()
					loss_rpn_box = rpn_loss_box.item()
					loss_rcnn_cls = RCNN_loss_cls.item()
					loss_rcnn_box = RCNN_loss_bbox.item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt

				print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (opt.session, epoch, step, iters_per_epoch, loss_temp, lr))
				print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

				if opt.use_tfboard:
					info = {
						'loss': loss_temp,
						'loss_rpn_cls': loss_rpn_cls,
						'loss_rpn_box': loss_rpn_box,
						'loss_rcnn_cls': loss_rcnn_cls,
						'loss_rcnn_box': loss_rcnn_box
					}
					logger.add_scalars("logs_s_{}/losses".format(opt.session), info, (epoch - 1) * iters_per_epoch + step)

				loss_temp = 0
				start = time.time()

		save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(opt.session, epoch, step))
		save_checkpoint({
			'session': opt.session,
			'epoch': epoch + 1,
			'model': fasterRCNN.module.state_dict() if opt.mGPUs else fasterRCNN.state_dict(),
			'optimizer': optimizer.state_dict(),
			'pooling_mode': cfg.POOLING_MODE,
			'class_agnostic': opt.class_agnostic,
		}, save_name)
		print('save model: {}'.format(save_name))

		if opt.use_tfboard:
			logger.close()

if __name__ == "__main__":
	main()