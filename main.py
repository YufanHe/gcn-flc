from __future__ import division
import argparse
import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from gcn_flc.flc_dataset import FlcDataset
from models.gcn import GCN

parser = argparse.ArgumentParser(description='GCN train and validate')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--learning_rate', type=float, default=0.0005, 
					help='learning rate (default: 0.005)')
parser.add_argument('--config', dest='config', default='config.json',
					help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--batch_size', type=int, default=2, 
					help='batch size (default: 2)')
parser.add_argument('--workers', type=int, default=4, 
					help='workers (default: 4)')
parser.add_argument('--no_cuda', action='store_true', default=True, 
					help='disables CUDA training')

def load_config(config_path):
	assert(os.path.exists(config_path))
	cfg = json.load(open(config_path, 'r'))
	return cfg

def build_data_loader(train_dataset):
	"""
	Build dataloader 
	Args: 
	Return: dataloader for training, validation 
	"""
	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle= True,
		num_workers=args.workers, pin_memory=False)
	# val_loader = DataLoader(
	#     val_dataset, batch_size=1, shuffle=False,
	#     num_workers=0, pin_memory=False)   
	# test_loader = DataLoader(
	#     test_loader, batch_size=1, shuffle=False,
	#     num_workers=0, pin_memory=False)     
	return train_loader #, val_loader
def train(cfg):

	device = torch.device("cuda" if not args.no_cuda else "cpu")

	dataset_path = './data/dataset.txt'
	train_dataset = FlcDataset(dataset_path, 200, split='train')

	train_loader = build_data_loader(train_dataset)
	print(cfg['network'])
	model = GCN(cfg['network']['input_feature'], cfg['network']['hidden_layer']).to(device)

	loss_fn = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)

	for epoch in range(args.epochs):

		model.train()

		for batch_count, (A, label, charge_weight, name) in enumerate(train_loader):

			input = torch.ones([args.batch_size, 200, cfg['network']['input_feature']], dtype=torch.float64)
			#cuda
			input, A, label = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float), label.to(device, dtype=torch.float)

			optimizer.zero_grad()

			output = model(input, A)

			print(output.shape)
			print(label.shape)


			loss = loss_fn(output, label)

			loss.backward()
			optimizer.step()

			print(loss.item())
def validate():
	pass
def main():
	global args
	args = parser.parse_args()
	cfg = load_config(args.config)

	train(cfg)


	
if __name__ == '__main__':
	main()

