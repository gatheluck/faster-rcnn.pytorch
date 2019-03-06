import os
import sys
import json

import torch
from torch import nn, optim

torch.backends.cudnn.benchmark=True

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)


__all__ = [ 
	'load_checkpoint', 
	'save_checkpoint', 
	'load_final', 
	'save_final',
	'save_result'
]

def load_checkpoint(model, optimizer, epoch, log_dir):
	model.load_state_dict(torch.load(os.path.join(log_dir, 'weight_{:03d}epoch.pth'.format(int(epoch)) )))
	optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'optimizer_{:03d}epoch.pth'.format(int(epoch)) )))


def save_checkpoint(model, optimizer, epoch, log_dir, mGPUs):
	if mGPUs:
		torch.save(model.module.state_dict(), os.path.join(log_dir, 'weight_{:03d}epoch.pth'.format(int(epoch)) ))
	else:
		torch.save(model.state_dict(), os.path.join(log_dir, 'weight_{:03d}epoch.pth'.format(int(epoch)) ))
	torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer_{:03d}epoch.pth'.format(int(epoch)) ))


def load_final(model, optimizer, log_dir):
	model.load_state_dict(torch.load(os.path.join(log_dir, 'weight_final.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'optimizer_final.pth')))


def save_final(model, optimizer, log_dir, mGPUs):
	if mGPUs:
		torch.save(model.module.state_dict(), os.path.join(log_dir, 'weight_final.pth'))
	else:
		torch.save(model.state_dict(), os.path.join(log_dir, 'weight_final.pth'))
	torch.save(optimizer.state_dict(), os.path.join(log_dir, 'optimizer_final.pth'))


def save_result(result, log_dir, filename):
	path = os.path.join(log_dir, filename)
	dir = os.path.dirname(path)
	os.makedirs(dir, exist_ok=True)

	with open(path, 'w') as f:
		f.write(json.dumps(result, indent=4))