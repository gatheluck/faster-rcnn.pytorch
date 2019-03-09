# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb

from options import TestOptions

try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3

datasets = ['pascal_voc']
architectures = ['vgg16','res50','res101','res152']

# def parse_args():
# 	"""
# 	Parse input arguments
# 	"""
# 	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
# 	parser.add_argument('--dataset', type=str, required=True, choices=datasets, help='training dataset')
# 	parser.add_argument('--net', type=str, required=True, choices=architectures, help='vgg16 | res50 | res101 | res152')
# 	parser.add_argument('--cfg', dest='cfg_file', type=str, required=True, help='optional config file')
# 	parser.add_argument('--set', dest='set_cfgs',
# 											help='set config keys', default=None,
# 											nargs=argparse.REMAINDER)
# 	# loading checkpoints from here 
# 	parser.add_argument('--load_dir', required=True, type=str, help='directory to load models')
# 	parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
# 	parser.add_argument('--ls', dest='large_scale',
# 											help='whether use large imag scale',
# 											action='store_true')
# 	parser.add_argument('--mGPUs', dest='mGPUs',
# 											help='whether use multiple GPUs',
# 											action='store_true')
# 	parser.add_argument('--cag', dest='class_agnostic',
# 											help='whether perform class_agnostic bbox regression',
# 											action='store_true')
# 	parser.add_argument('--parallel_type', dest='parallel_type',
# 											help='which part of model to parallel, 0: all, 1: model before roi pooling',
# 											default=0, type=int)
# 	parser.add_argument('--checksession', dest='checksession',
# 											help='checksession to load model',
# 											default=1, type=int)
# 	parser.add_argument('--checkepoch', dest='checkepoch',
# 											help='checkepoch to load network',
# 											default=1, type=int)
# 	parser.add_argument('--checkpoint', dest='checkpoint',
# 											help='checkpoint to load network',
# 											default=10021, type=int)
# 	parser.add_argument('--vis', dest='vis',
# 											help='visualization mode',
# 											action='store_true')
# 	args = parser.parse_args()
# 	return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def test_net(opt):
	# if opt == None:
	# 	opt = TestOptions().parse()
	# else:
	lr = opt.lr
	momentum = cfg.TRAIN.MOMENTUM
	weight_decay = opt.wd

	print('Called with opt:')
	print(opt)

	# if torch.cuda.is_available() and not args.cuda:
	# 	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	np.random.seed(cfg.RNG_SEED)
	if opt.dataset == "pascal":
		opt.imdb_name = "voc_2007_trainval"
		opt.imdbval_name = "voc_2007_test"
		opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	# elif args.dataset == "pascal_voc_0712":
	# 		args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
	# 		args.imdbval_name = "voc_2007_test"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	# elif args.dataset == "coco":
	# 		args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
	# 		args.imdbval_name = "coco_2014_minival"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	# elif args.dataset == "imagenet":
	# 		args.imdb_name = "imagenet_train"
	# 		args.imdbval_name = "imagenet_val"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	# elif args.dataset == "vg":
	# 		args.imdb_name = "vg_150-50-50_minitrain"
	# 		args.imdbval_name = "vg_150-50-50_minival"
	# 		args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	else:
		raise NotImplementedError

	# opt.cfg_file = "cfgs/{}_ls.yml".format(opt.net) if opt.large_scale else "cfgs/{}.yml".format(opt.net)
	opt.cfg_file = "cfgs/{}.yml".format(opt.arch)

	if opt.cfg_file is not None:
		cfg_from_file(opt.cfg_file)
	if opt.set_cfgs is not None:
		cfg_from_list(opt.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)

	cfg.TRAIN.USE_FLIPPED = False
	imdb, roidb, ratio_list, ratio_index = combined_roidb(opt.imdbval_name, False)
	imdb.competition_mode(on=True)

	print('{:d} roidb entries'.format(len(roidb)))

	# input_dir = opt.load_dir + "/" + opt.net + "/" + opt.dataset
	# if not os.path.exists(input_dir):
	# 	raise Exception('There is no input directory for loading network from ' + input_dir)
	load_name = os.path.join(opt.log_dir,'checkpoint.pth')

	# initilize the network here.
	if opt.arch == 'vgg16':
		fasterRCNN = vgg16(imdb.classes, bb_weight=None, class_agnostic=opt.cag, batch_norm=False)
	elif opt.arch == 'vgg16bn':
		fasterRCNN = vgg16(imdb.classes, bb_weight=None, class_agnostic=opt.cag, batch_norm=True)
	elif opt.arch == 'resnet101':
		fasterRCNN = resnet(imdb.classes, 101, bb_weight=None, class_agnostic=opt.cag)
	elif opt.arch == 'resnet50':
		fasterRCNN = resnet(imdb.classes, 50, bb_weight=None, class_agnostic=opt.cag)
	elif opt.arch == 'resnet152':
		fasterRCNN = resnet(imdb.classes, 152, bb_weight=None, class_agnostic=opt.cag)
	else:
		print("network is not defined")
		pdb.set_trace()

	fasterRCNN.create_architecture()

	print("load checkpoint %s" % (load_name))
	checkpoint = torch.load(load_name)
	fasterRCNN.load_state_dict(checkpoint['model'])
	if 'pooling_mode' in checkpoint.keys():
		cfg.POOLING_MODE = checkpoint['pooling_mode']


	print('load model successfully!')
	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1).to(opt.device)
	im_info = torch.FloatTensor(1).to(opt.device)
	num_boxes = torch.LongTensor(1).to(opt.device)
	gt_boxes = torch.FloatTensor(1).to(opt.device)

	# ship to cuda
	# if args.cuda:
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

	if opt.cuda:
		fasterRCNN.cuda()

	start = time.time()
	max_per_image = 100

	vis = False # for debugging

	if vis:
		thresh = 0.05
	else:
		thresh = 0.0

	# save_name = 'faster_rcnn_10'
	num_images = len(imdb.image_index)
	# if opt.debugging: num_images = 100
	all_boxes = [[[] for _ in xrange(num_images)]
								for _ in xrange(imdb.num_classes)]

	#output_dir = get_output_dir(imdb, save_name)
	output_dir = opt.log_dir+'/test_result'
	os.makedirs(output_dir, exist_ok=True)

	dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=False, normalize = False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers, pin_memory=True)

	data_iter = iter(dataloader)

	_t = {'im_detect': time.time(), 'misc': time.time()}
	det_file = os.path.join(output_dir, 'detections.pkl')

	fasterRCNN.eval()
	empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
	for i in range(num_images):

		data = next(data_iter)
		im_data.data.resize_(data[0].size()).copy_(data[0])
		im_info.data.resize_(data[1].size()).copy_(data[1])
		gt_boxes.data.resize_(data[2].size()).copy_(data[2])
		num_boxes.data.resize_(data[3].size()).copy_(data[3])

		det_tic = time.time()
		rois, cls_prob, bbox_pred, \
		rpn_loss_cls, rpn_loss_box, \
		RCNN_loss_cls, RCNN_loss_bbox, \
		rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

		scores = cls_prob.data
		boxes = rois.data[:, :, 1:5]

		if cfg.TEST.BBOX_REG:
			# Apply bounding-box regression deltas
			box_deltas = bbox_pred.data
			if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
			# Optionally normalize targets by a precomputed mean and stdev
				if opt.cag:
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
											+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					box_deltas = box_deltas.view(1, -1, 4)
				else:
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
											+ torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

			pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
			pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
		else:
			# Simply repeat the boxes, once for each class
			pred_boxes = np.tile(boxes, (1, scores.shape[1]))

		pred_boxes /= data[1][0][2].item()

		scores = scores.squeeze()
		pred_boxes = pred_boxes.squeeze()
		det_toc = time.time()
		detect_time = det_toc - det_tic
		misc_tic = time.time()
		if vis:
			im = cv2.imread(imdb.image_path_at(i))
			im2show = np.copy(im)
		for j in xrange(1, imdb.num_classes):
			inds = torch.nonzero(scores[:,j]>thresh).view(-1)
			# if there is det
			if inds.numel() > 0:
				cls_scores = scores[:,j][inds]
				_, order = torch.sort(cls_scores, 0, True)
				if opt.cag:
					cls_boxes = pred_boxes[inds, :]
				else:
					cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
				
				cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
				# cls_dets = torch.cat((cls_boxes, cls_scores), 1)
				cls_dets = cls_dets[order]
				keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
				cls_dets = cls_dets[keep.view(-1).long()]
				if vis:
					im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
				all_boxes[j][i] = cls_dets.cpu().numpy()
			else:
				all_boxes[j][i] = empty_array

		# Limit to max_per_image detections *over all classes*
		if max_per_image > 0:
			image_scores = np.hstack([all_boxes[j][i][:, -1]
																for j in xrange(1, imdb.num_classes)])
			if len(image_scores) > max_per_image:
				image_thresh = np.sort(image_scores)[-max_per_image]
				for j in xrange(1, imdb.num_classes):
					keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
					all_boxes[j][i] = all_boxes[j][i][keep, :]

		misc_toc = time.time()
		nms_time = misc_toc - misc_tic

		sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(i + 1, num_images, detect_time, nms_time))
		sys.stdout.flush()

		if vis:
			cv2.imwrite('result.png', im2show)
			#pdb.set_trace()
			#cv2.imshow('test', im2show)
			#cv2.waitKey(0)

	with open(det_file, 'wb') as f:
		pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

	print('Evaluating detections')
	imdb.evaluate_detections(all_boxes, output_dir, opt.log_dir.split('/')[-1]+'.json')

	end = time.time()
	print("test time: %0.4fs" % (end - start))

if __name__ == '__main__':
	opt = TestOptions().parse()
	test_net(opt)