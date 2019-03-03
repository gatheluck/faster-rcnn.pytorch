from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

from ESLSA.models.classifier import *
from ESLSA.models import resnet

class resnet(_fasterRCNN):
	def __init__(self, classes, num_layers, opt, class_agnostic=False):
		self.model_path = opt.weight_bb
		self.opt = opt
		self.class_agnostic = class_agnostic

		if (self.opt.model == 'resnet18') or (self.opt.model == 'resnet34'):
			self.dout_base_model = 256			
		elif self.opt.model == 'resnet50':
			self.dout_base_model = 1024
		else:
			raise NotImplementedError

		_fasterRCNN.__init__(self, classes, class_agnostic)

	def _init_modules(self):
		if self.opt.use_IN_pretrained_weight == True:
			print("Loading ImageNet pretrained weights")
			resnet = get_classifier(self.opt.model, self.opt.num_class_bb, self.opt.application, self.opt.activation, self.opt.beta, inplace=True, pretrained=True)
		else:
			print("Loading pretrained weights from %s" %(self.model_path))
			state_dict = torch.load(self.model_path)
			resnet = get_classifier(self.opt.model, self.opt.num_class_bb, self.opt.application, self.opt.activation, self.opt.beta, inplace=True, pretrained=False)
			#resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
			resnet.load_state_dict(torch.load(self.model_path))

		# RCNN
		if (self.opt.model == 'resnet18') or (self.opt.model == 'resnet34') or (self.opt.model == 'resnet50') or (self.opt.model =='resnet101'):
			self.RCNN_base = nn.Sequential(resnet.features._modules["0"], 
																		resnet.features._modules["1"],
																		resnet.features._modules["2"],
																		resnet.features._modules["3"],
																		resnet.features._modules["4"],
																		resnet.features._modules["5"],
																		resnet.features._modules["6"])
			self.RCNN_top = nn.Sequential(resnet.features._modules["7"])
		else:
			raise NotImplementedError


		if (self.opt.model == 'resnet18') or (self.opt.model == 'resnet34'):
			self.RCNN_cls_score = nn.Linear(512, self.n_classes)
			if self.class_agnostic:
				self.RCNN_bbox_pred = nn.Linear(512, 4)
			else:
				self.RCNN_bbox_pred = nn.Linear(512, 4 * self.n_classes)
		
		elif self.opt.model == 'resnet50':
			self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
			if self.class_agnostic:
				self.RCNN_bbox_pred = nn.Linear(2048, 4)
			else:
				self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

		else:
			raise NotImplementedError

		# Fix blocks
		for p in self.RCNN_base[0].parameters(): p.requires_grad=False
		for p in self.RCNN_base[1].parameters(): p.requires_grad=False

		assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
		if cfg.RESNET.FIXED_BLOCKS >= 3:
			for p in self.RCNN_base[6].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 2:
			for p in self.RCNN_base[5].parameters(): p.requires_grad=False
		if cfg.RESNET.FIXED_BLOCKS >= 1:
			for p in self.RCNN_base[4].parameters(): p.requires_grad=False

		def set_bn_fix(m):
			classname = m.__class__.__name__
			if classname.find('BatchNorm') != -1:
				for p in m.parameters(): p.requires_grad=False

		self.RCNN_base.apply(set_bn_fix)
		self.RCNN_top.apply(set_bn_fix)

	def train(self, mode=True):
		# Override train so that the training mode is set as we want
		nn.Module.train(self, mode)
		if mode:
			# Set fixed blocks to be in eval mode
			self.RCNN_base.eval()
			self.RCNN_base[5].train()
			self.RCNN_base[6].train()

			def set_bn_eval(m):
				classname = m.__class__.__name__
				if classname.find('BatchNorm') != -1:
					m.eval()

			self.RCNN_base.apply(set_bn_eval)
			self.RCNN_top.apply(set_bn_eval)

	def _head_to_tail(self, pool5):
		fc7 = self.RCNN_top(pool5).mean(3).mean(2)
		return fc7