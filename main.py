from __future__ import division
import argparse
import os
import numpy as np
import json

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

from gcn_flc.flc_dataset import FlcDataset
from models.gcn import GCN

parser = argparse.ArgumentParser(description='GCN train and validate')
parser.add_argument('--mode', default='train', choices=['train', 'predict'], 
					help='operation mode: train or predict (default: train)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--learning_rate', type=float, default=0.005, 
					help='learning rate (default: 0.005)')
parser.add_argument('--config', dest='config', default='config.json',
					help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--batch_size', type=int, default=4, 
					help='batch size (default: 4)')
parser.add_argument('--workers', type=int, default=4, 
					help='workers (default: 4)')
parser.add_argument('--no_cuda', action='store_true', default=True, 
					help='disables CUDA training')

def load_config(config_path):
	assert(os.path.exists(config_path))
	cfg = json.load(open(config_path, 'r'))
	return cfg

def build_data_loader():
	"""
	Build dataloader 
	Args: 
	Return: dataloader for training, validation 
	"""
	dataset_path = './dataset/synthetic/dataset_100.json'

	train_dataset = FlcDataset(dataset_path, split='train')
	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle= True,
		num_workers=args.workers, pin_memory=False)

	val_dataset = FlcDataset(dataset_path, split='val')
	val_loader = DataLoader(
	    val_dataset, batch_size=1, shuffle=False,
	    num_workers=0, pin_memory=False)   
	# test_loader = DataLoader(
	#     test_loader, batch_size=1, shuffle=False,
	#     num_workers=0, pin_memory=False)     
	return train_loader, val_loader
def train(cfg, model, train_loader, device, optimizer, loss_list):

	model.train()

	for batch_count, (A, label, charge_weight, index) in enumerate(train_loader):

		input = torch.ones([args.batch_size, cfg['data']['total_nodes'],
				 			cfg['network']['input_feature']], dtype=torch.float64)
		#cuda
		input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
		label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)
		
		label.unsqueeze_(-1)
		charge_weight.unsqueeze_(-1)

		optimizer.zero_grad()

		output = model(input, A)
		
		loss = F.binary_cross_entropy(output, label, weight=charge_weight)

		loss.backward()
		optimizer.step()

		loss_list.append(loss.item())
		

def validate():
	pass

def main():
	global args
	args = parser.parse_args()
	cfg = load_config(args.config)

	device = torch.device("cuda" if not args.no_cuda else "cpu")
	
	if args.mode == 'train':

		train_loader, val_loader = build_data_loader()

		model = GCN(cfg['network']).to(device)

		optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)

		loss_list = []

		for epoch in range(args.epochs):

			train(cfg, model, train_loader, device, optimizer, loss_list)

		plt.plot(loss_list)
		plt.show()

	elif args.mode == 'predict':
		pass

	else:
		raise Exception("Unrecognized mode.")
	
	
	
if __name__ == '__main__':
	main()

