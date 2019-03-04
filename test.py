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
from model.faster_rcnn.resnet import resnet

import pdb
from option import TestOptions

try:
	xrange          # Python 2
except NameError:
	xrange = range  # Python 3


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def main():
	opt = TestOptions().parse()

	np.random.seed(cfg.RNG_SEED)
	if opt.dataset_rcnn == "pascal_voc":
		opt.imdb_name = "voc_2007_trainval"
		opt.imdbval_name = "voc_2007_test"
		opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	elif opt.dataset_rcnn == "pascal_voc_0712":
		opt.imdb_name = "voc_2007_trainval+voc_2012_trainval"
		opt.imdbval_name = "voc_2007_test"
		opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	elif opt.dataset_rcnn == "coco":
		opt.imdb_name = "coco_2014_train+coco_2014_valminusminival"
		opt.imdbval_name = "coco_2014_minival"
		opt.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	elif opt.dataset_rcnn == "imagenet":
		opt.imdb_name = "imagenet_train"
		opt.imdbval_name = "imagenet_val"
		opt.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
	elif opt.dataset_rcnn == "vg":
		opt.imdb_name = "vg_150-50-50_minitrain"
		opt.imdbval_name = "vg_150-50-50_minival"
		opt.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

	opt.cfg_file = "cfgs/{}_ls.yml".format(opt.model) if opt.large_scale else "cfgs/{}.yml".format(opt.model)

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

	# input_dir = opt.logs + "/" + opt.model + "/" + opt.dataset
	# if not os.path.exists(input_dir):
	# 	raise Exception('There is no input directory for loading network from ' + input_dir)
	# load_name = os.path.join(input_dir,'faster_rcnn_{}_{}_{}.pth'.format(opt.checksession, opt.checkepoch, opt.checkpoint))

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

	print("load checkpoint %s" % (opt.weight_rcnn))
	checkpoint = torch.load(opt.weight_rcnn)
	fasterRCNN.load_state_dict(checkpoint['model'])
	fasterRCNN = fasterRCNN.to(opt.device)

	if 'pooling_mode' in checkpoint.keys():
		cfg.POOLING_MODE = checkpoint['pooling_mode']

	print('load model successfully!')
	# initilize the tensor holder here.
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

	start = time.time()
	max_per_image = 100

	vis = opt.vis

	if vis:
		thresh = 0.05
	else:
		thresh = 0.0

	num_images = len(imdb.image_index)
	all_boxes = [[[] for _ in xrange(num_images)]
								for _ in xrange(imdb.num_classes)]

	dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=False, normalize = False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

	data_iter = iter(dataloader)

	_t = {'im_detect': time.time(), 'misc': time.time()}
	det_file = os.path.join(opt.log_dir, 'detections.pkl')
	
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
				if opt.class_agnostic:
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
				if opt.class_agnostic:
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

		sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
			.format(i + 1, num_images, detect_time, nms_time))
		sys.stdout.flush()

		if vis:
			cv2.imwrite(opt.log_dir+'/result{:04d}.png'.format(i), im2show)
			# pdb.set_trace()
			#cv2.imshow('test', im2show)
			#cv2.waitKey(0)


	with open(det_file, 'wb') as f:
		pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

	print('Evaluating detections')
	imdb.evaluate_detections(all_boxes, opt.log_dir)

	end = time.time()
	print("test time: %0.4fs" % (end - start))



if __name__ == "__main__":
	main()