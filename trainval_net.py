# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from tqdm import tqdm

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
from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, clip_gradient, save_checkpoint
#from model.utils.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient

from tensorboardX import SummaryWriter

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from options import TrainOptions
from misc import save_weight, save_final, save_result
# from test_net import test

# datasets = ['pascal_voc']
# architectures = ['vgg16','res50','res101','res152']

# def parse_args():
# 	"""
# 	Parse input arguments
# 	"""
# 	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
# 	parser.add_argument('--dataset', type=str, required=True, choices=datasets, help='training dataset')
# 	parser.add_argument('--net', type=str, required=True, choices=architectures, help='vgg16 | res50 | res101 | res152')
# 	parser.add_argument('--bb_weight', type=str, required=True, help='path to backbone weight')
# 	# parser.add_argument('--start_epoch', dest='start_epoch',
# 	# 										help='starting epoch',
# 	# 										default=1, type=int)
# 	parser.add_argument('--epochs', dest='max_epochs',
# 											help='number of epochs to train',
# 											default=20, type=int)
# 	parser.add_argument('--disp_interval', dest='disp_interval',
# 											help='number of iterations to display',
# 											default=100, type=int)
# 	parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
# 											help='number of iterations to display',
# 											default=10000, type=int)

# 	parser.add_argument('--save_dir', dest='save_dir', type=str, required=True, help='directory to save models')
# 	parser.add_argument('--nw', dest='num_workers',
# 											help='number of worker to load data',
# 											default=0, type=int)
# 	parser.add_argument('--cuda', dest='cuda',
# 											help='whether use CUDA',
# 											action='store_true')
# 	parser.add_argument('--ls', dest='large_scale',
# 											help='whether use large imag scale',
# 											action='store_true')                      
# 	parser.add_argument('--mGPUs', dest='mGPUs',
# 											help='whether use multiple GPUs',
# 											action='store_true')
# 	parser.add_argument('--bs', dest='batch_size', type=int, required=True, help='batch_size')
# 	parser.add_argument('--cag', dest='class_agnostic',
# 											help='whether perform class_agnostic bbox regression',
# 											action='store_true')

# 	# config optimization
# 	parser.add_argument('--o', dest='optimizer',
# 											help='training optimizer',
# 											default="sgd", type=str)
# 	parser.add_argument('--lr', dest='lr',
# 											help='starting learning rate',
# 											default=0.001, type=float)
# 	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
# 											help='step to do learning rate decay, unit is epoch',
# 											default=5, type=int)
# 	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
# 											help='learning rate decay ratio',
# 											default=0.1, type=float)

	# set training session
	# parser.add_argument('--s', dest='session',
	# 										help='training session',
	# 										default=1, type=int)

	# resume trained model
	# parser.add_argument('--r', dest='resume',
	# 										help='resume checkpoint or not',
	# 										default=False, type=bool)
	# parser.add_argument('--checksession', dest='checksession',
	# 										help='checksession to load model',
	# 										default=1, type=int)
	# parser.add_argument('--checkepoch', dest='checkepoch',
	# 										help='checkepoch to load model',
	# 										default=1, type=int)
	# parser.add_argument('--checkpoint', dest='checkpoint',
	# 										help='checkpoint to load model',
	# 										default=0, type=int)
	# log and diaplay
	# parser.add_argument('--use_tfb', dest='use_tfboard',
	# 										help='whether use tensorboard',
	# 										action='store_true')

	# args = parser.parse_args()
	# return args

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



	# args = parse_args()

	# print('Called with args:')
	# print(args)

def trainval_net(opt):

	if opt.dataset == "pascal":
			opt.imdb_name    = "voc_2007_trainval"
			opt.imdbval_name = "voc_2007_test"
			opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	# elif args.dataset == "pascal_voc_0712":
	# 		args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
	# 		args.imdbval_name = "voc_2007_test"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	# elif args.dataset == "coco":
	# 		args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
	# 		args.imdbval_name = "coco_2014_minival"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	# elif args.dataset == "imagenet":
	# 		args.imdb_name = "imagenet_train"
	# 		args.imdbval_name = "imagenet_val"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	# elif args.dataset == "vg":
	# 		# train sizes: train, smalltrain, minitrain
	# 		# train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
	# 		args.imdb_name = "vg_150-50-50_minitrain"
	# 		args.imdbval_name = "vg_150-50-50_minival"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	else:
		raise NotImplementedError


	# if args.large_scale:
	# 	args.cfg_file = "cfgs/{}_ls.yml".format(args.net) 
	# else:
	opt.cfg_file = "cfgs/{}.yml".format(opt.arch)

	# args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

	if opt.cfg_file is not None:
		cfg_from_file(opt.cfg_file)
	if opt.set_cfgs is not None:
		cfg_from_list(opt.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	# if torch.cuda.is_available() and not args.cuda:
	# 	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# train set
	# -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = opt.cuda
	imdb, roidb, ratio_list, ratio_index = combined_roidb(opt.imdb_name)
	train_size = len(roidb)
	# if opt.debugging : train_size=100 # for debugging

	print('{:d} roidb entries'.format(len(roidb)))

	# if not os.path.exists(out.log_dir):
	# 	os.makedirs(output_dir)

	sampler_batch = sampler(train_size, opt.batch_size)

	dataset = roibatchLoader(roidb, ratio_list, ratio_index, opt.batch_size, imdb.num_classes, training=True)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler_batch, num_workers=opt.num_workers)

	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1).to(opt.device)
	im_info = torch.FloatTensor(1).to(opt.device)
	num_boxes = torch.LongTensor(1).to(opt.device)
	gt_boxes = torch.FloatTensor(1).to(opt.device)

	# ship to cuda
	# if opt.cuda:
	# 	im_data = im_data.cuda()
	# 	im_info = im_info.cuda()
	# 	num_boxes = num_boxes.cuda()
	# 	gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if opt.cuda:
		cfg.CUDA = True

	# initilize the network here.
	if opt.arch == 'vgg16':
		fasterRCNN = vgg16(imdb.classes, opt.bb_weight, class_agnostic=opt.cag, batch_norm=False)
	elif opt.arch == 'vgg16bn':
		fasterRCNN = vgg16(imdb.classes, opt.bb_weight, class_agnostic=opt.cag, batch_norm=True)
	elif opt.arch == 'resnet50':
		fasterRCNN = resnet(imdb.classes, 50, opt.bb_weight, class_agnostic=opt.cag)
	elif opt.arch == 'resnet101':
		fasterRCNN = resnet(imdb.classes, 101, opt.bb_weight, class_agnostic=opt.cag)
	elif opt.arch == 'resnet152':
		fasterRCNN = resnet(imdb.classes, 152, opt.bb_weight, class_agnostic=opt.cag)
	else:
		print("network is not defined")
		pdb.set_trace()

	fasterRCNN.create_architecture()

	lr = cfg.TRAIN.LEARNING_RATE
	lr = opt.lr
	#tr_momentum = cfg.TRAIN.MOMENTUM
	#tr_momentum = args.momentum

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params':[value],'lr':lr, 'weight_decay': opt.wd}]
				# params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
				# 				'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
			else:
				params += [{'params':[value],'lr':lr, 'weight_decay': opt.wd}]

	if opt.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params)

	elif opt.optimizer == "sgd":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	# if args.resume:
	# 	load_name = os.path.join(output_dir,
	# 		'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
	# 	print("loading checkpoint %s" % (load_name))
	# 	checkpoint = torch.load(load_name)
	# 	args.session = checkpoint['session']
	# 	args.start_epoch = checkpoint['epoch']
	# 	fasterRCNN.load_state_dict(checkpoint['model'])
	# 	optimizer.load_state_dict(checkpoint['optimizer'])
	# 	lr = optimizer.param_groups[0]['lr']
	# 	if 'pooling_mode' in checkpoint.keys():
	# 		cfg.POOLING_MODE = checkpoint['pooling_mode']
	# 	print("loaded checkpoint %s" % (load_name))

	if opt.mGPUs:	fasterRCNN = nn.DataParallel(fasterRCNN)
	if opt.cuda:	fasterRCNN.cuda()

	iters_per_epoch = int(train_size / opt.batch_size)

	# tensor board
	logger = SummaryWriter(opt.log_dir+"/runs")

	print("cfg.POOLING_MODE:", cfg.POOLING_MODE)

	# epoch loop
	# for epoch in range(args.start_epoch, args.max_epochs + 1):
	for epoch in range(1, opt.num_epochs + 1):
		# setting to train mode
		fasterRCNN.train()
		loss_temp = 0
		start = time.time()

		if epoch % (opt.step_size + 1) == 0:
			adjust_learning_rate(optimizer, opt.gamma)
			lr *= opt.wd

		data_iter = iter(dataloader)

		# step loop
		# iters_per_epoch = 2 # for debug
		# for step in range(iters_per_epoch):
		for step in tqdm(range(iters_per_epoch)):
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

			loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
						+ RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
			loss_temp += loss.item()

			# backward
			optimizer.zero_grad()
			loss.backward()
			if opt.arch == "vgg16":
					clip_gradient(fasterRCNN, 10.)
			optimizer.step()

			if step % opt.print_freq == 0:
				end = time.time()
				if step > 0:
					loss_temp /= (opt.print_freq + 1)

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

				# print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
				# print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				# print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

				print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (epoch, step, iters_per_epoch, loss_temp, lr))
				print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))


				# tensorboard
				info = {
					'loss': loss_temp,
					'loss_rpn_cls': loss_rpn_cls,
					'loss_rpn_box': loss_rpn_box,
					'loss_rcnn_cls': loss_rcnn_cls,
					'loss_rcnn_box': loss_rcnn_box
				}
				logger.add_scalars("losses", info, (epoch - 1) * iters_per_epoch + step)

				loss_temp = 0
				start = time.time()

		
		# save_name = os.path.join(opt.log_dir, 'faster_rcnn_{}_{}.pth'.format(epoch, step))
		if epoch % opt.checkpoint == 0:
			# save_name = os.path.join(opt.log_dir, '{}_{}_{}_{:03d}epoch_{:06d}step.pth'.format(opt.arch, opt.dataset, opt.bb_weight.split("_")[-1], epoch, step))
			checkpoint_name = os.path.join(opt.log_dir, 'checkpoint.pth')
			save_checkpoint({
				'epoch': epoch + 1,
				'model': fasterRCNN.module.state_dict() if opt.mGPUs else fasterRCNN.state_dict(),
				'optimizer': optimizer.state_dict(),
				'pooling_mode': cfg.POOLING_MODE,
				'class_agnostic': opt.cag,
			}, checkpoint_name)

			weight_name = os.path.join(opt.log_dir, 'weight_{:03d}epoch.pth'.format(epoch))
			save_weight(fasterRCNN, epoch, opt.log_dir, opt.mGPUs)
			print('save weight: {}'.format(save_weight))


	checkpoint_name = os.path.join(opt.log_dir, 'checkpoint.pth')
	save_checkpoint({
		'epoch': epoch + 1,
		'model': fasterRCNN.module.state_dict() if opt.mGPUs else fasterRCNN.state_dict(),
		'optimizer': optimizer.state_dict(),
		'pooling_mode': cfg.POOLING_MODE,
		'class_agnostic': opt.cag,
	}, checkpoint_name)
	save_final(fasterRCNN, optimizer, opt.log_dir, opt.mGPUs)	
	save_result({
		'loss': loss_temp,
		'loss_rpn_cls': loss_rpn_cls,
		'loss_rpn_box': loss_rpn_box,
		'loss_rcnn_cls': loss_rcnn_cls,
		'loss_rcnn_box': loss_rcnn_box
	}, opt.log_dir, opt.result)

	logger.close()

	# run test
	# if not opt.train_only: test(opt)

if __name__ == "__main__":
	opt = TrainOptions().parse()
	trainval_net(opt)