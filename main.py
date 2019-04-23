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
parser.add_argument('--epochs', default=500, type=int, metavar='N',
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
	train_dataset_path = './dataset/synthetic/dataset_500.json'
	val_dataset_path = './dataset/synthetic/dataset_100.json'

	train_dataset = FlcDataset(train_dataset_path, split='train')
	train_loader = DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle= True,
		num_workers=args.workers, pin_memory=False)

	val_dataset = FlcDataset(val_dataset_path, split='val')
	val_loader = DataLoader(
		val_dataset, batch_size=1, shuffle=False,
		num_workers=0, pin_memory=False)   
	# test_loader = DataLoader(
	#     test_loader, batch_size=1, shuffle=False,
	#     num_workers=0, pin_memory=False)     
	return train_loader, val_loader
def train(cfg, model, train_loader, device, optimizer, epoch, train_loss_list):

	model.train()

	for batch_count, (A, label, charge_weight, f_num, index) in enumerate(train_loader):

		input = torch.ones([args.batch_size, cfg['data']['total_nodes'],
							cfg['network']['input_feature']], dtype=torch.float64)

		#input = charge_weight
		#cuda
		input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
		label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)

		optimizer.zero_grad()

		output = model(input, A)
		
		#loss = F.binary_cross_entropy(output, label, weight=charge_weight)
		loss = F.binary_cross_entropy(output, label)

		loss.backward()
		optimizer.step()

		train_loss_list.append(loss.item())

		if batch_count%10 == 0:
			print('Epoch [%d/%d] Loss: %.4f' %(epoch, args.epochs, loss.item()))
		

def validate(cfg, model, val_loader, device, val_loss_list):
	
	model.eval()

	val_loss = 0

	with torch.no_grad():

		for batch_count, (A, label, charge_weight, f_num, index) in enumerate(val_loader):

			input = torch.ones([1, cfg['data']['total_nodes'],
							cfg['network']['input_feature']], dtype=torch.float64)
			#input = charge_weight

			#cuda
			input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
			label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)

			output = model(input, A)

			#loss = F.binary_cross_entropy(output, label, weight=charge_weight)
			loss = F.binary_cross_entropy(output, label)

			val_loss += loss.item()

		avg_loss = val_loss / len(val_loader.dataset)
		val_loss_list.append(avg_loss)

		print('##Validate loss : %.4f' %(avg_loss))

def test(cfg, model, val_loader, device):

	model.eval()

	val_loss = 0

	with torch.no_grad():

		for batch_count, (A, label, charge_weight, f_num, index) in enumerate(val_loader):

			input = torch.ones([1, cfg['data']['total_nodes'],
							cfg['network']['input_feature']], dtype=torch.float64)

			#cuda
			input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
			label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)

			output = model(input, A)

			print(output[:, :f_num+2, :])
			print(label[:, :f_num +2, :])

def main():
	global args
	args = parser.parse_args()
	cfg = load_config(args.config)

	device = torch.device("cuda" if not args.no_cuda else "cpu")
	
	if args.mode == 'train':

		train_loader, val_loader = build_data_loader()

		model = GCN(cfg).to(device)

		optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)

		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

		train_loss_list = []
		val_loss_list = []

		for epoch in range(args.epochs):
			scheduler.step(epoch)

			train(cfg, model, train_loader, device, optimizer, epoch, train_loss_list)
			validate(cfg, model, val_loader, device, val_loss_list)

		test(cfg, model, val_loader, device)
		
		plt.figure(1)
		plt.plot(train_loss_list)
		plt.figure(2)
		plt.plot(val_loss_list)
		plt.show()
		
	elif args.mode == 'predict':
		pass

	else:
		raise Exception("Unrecognized mode.")
	
	
	
if __name__ == '__main__':
	main()

