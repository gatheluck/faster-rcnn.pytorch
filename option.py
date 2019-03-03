import os
import argparse
import json
import torch
import torch.nn as nn

__all__ = [
	'TrainOptions',
	'TestOptions',
]

class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('--dataset_rcnn', type=str, help='training dataset of rcnn', default='pascal_voc')
		parser.add_argument('--dataset_bb', type=str, default='cifar10', help='cifar10 | imagenet')
		parser.add_argument('--model', type=str, required=True, help='lenet | alexnet | resnet18 | resnet34')
		parser.add_argument('--weight_bb', type=str, required=True, help='path of backborn model weight')
		parser.add_argument('--use_IN_pretrained_weight', action='store_true', default=False, help='use official ImageNet pretrained weight')
		parser.add_argument('--application', type=str, default='all', help='all | fc')
		parser.add_argument('--activation', type=str, default='relu', help='relu | softplus | sigmoid | tanh')
		parser.add_argument('--beta', type=float, default=100.0, help='beta value when activation type is softplus')
		parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
		parser.add_argument('--epochs', dest='max_epochs', type=int, default=20, help='number of epochs to train')
		parser.add_argument('--disp_interval', type=int, default=100, help='number of iterations to display')
		parser.add_argument('--checkpoint_interval', type=int, default=10000, help='number of iterations to display')

		parser.add_argument('--log_dir', type=str, required=True, help='directory to save models')
		parser.add_argument('--nw', dest='num_workers', type=int, default=0, help='number of worker to load data')
		parser.add_argument('--use_gpu', action='store_true', default=False, help='GPU mode ON/OFF')
		parser.add_argument('--ls', dest='large_scale', action='store_true', help='whether use large imag scale')
		parser.add_argument('--mGPUs', dest='mGPUs', action='store_true', help='whether use multiple GPUs')
		parser.add_argument('--bs', dest='batch_size', default=1, type=int, help='batch_size')
		parser.add_argument('--cag', dest='class_agnostic', action='store_true',help='whether perform class_agnostic bbox regression')

		# optimization
		parser.add_argument('--o', dest='optimizer', type=str, default="sgd", help='training optimizer')
		parser.add_argument('--lr', type=float, default=0.001, help='starting learning rate')
		parser.add_argument('--lr_decay_step', type=int, default=5, help='step to do learning rate decay, unit is epoch')
		parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='learning rate decay ratio')
		# training session
		parser.add_argument('--s', dest='session', type=int, default=1, help='training session')

		# resume trained model
		parser.add_argument('--r', dest='resume', type=bool, default=False, help='resume checkpoint or not')
		parser.add_argument('--checksession', dest='checksession', type=int, default=1, help='checksession to load model')
		parser.add_argument('--checkepoch', dest='checkepoch', type=int, default=1, help='checkepoch to load model')
		parser.add_argument('--checkpoint', dest='checkpoint', type=int, default=0, help='checkpoint to load model')
		# tebsor board
		parser.add_argument('--use_tfb', dest='use_tfboard', action='store_true', help='whether use tensorboard')

		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)
			
		self.parser = parser
		return parser.parse_args()
  
	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')
  
	def parse(self):
		opt = self.gather_options()
		self.print_options(opt)

		if opt.model == 'lenet':
			opt.input_size = 32
		else:
			opt.input_size = 224

		# get number of categories
		if opt.use_IN_pretrained_weight:
			opt.num_class_bb = 1000
		else:
			if opt.dataset_bb == 'cifar10':
				opt.num_class_bb = 10
			elif opt.dataset_bb == 'imagenet':
				opt.num_class_bb = 1000
			else:
				raise NotImplementedError

		# GPU
		if opt.use_gpu and torch.cuda.is_available():
			torch.backends.cudnn.benchmark = True
			opt.device = 'cuda'
		else:
			opt.device = 'cpu'

		self.opt = opt
		return self.opt


class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		return parser