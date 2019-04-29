from __future__ import division
import argparse
import os
import numpy as np
import json

from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchsummary import summary

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
parser.add_argument('--learning_rate', type=float, default=0.0005, 
					help='learning rate (default: 0.005)')
parser.add_argument('--config', dest='config', default='config.json',
					help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--batch_size', type=int, default=4, 
					help='batch size (default: 4)')
parser.add_argument('--workers', type=int, default=4, 
					help='workers (default: 4)')
parser.add_argument('--no_cuda', action='store_true', default=True, 
					help='disables CUDA training')
parser.add_argument('--resume_file', type=str, default='/home/kuowei/Desktop/gcn-flc/checkpoint/gcn_wc1_small_bs_4_lr_5e-4_model.pkl', 
					help='the checkpoint file to resume from')

tb_log_dir = './tb_log/'
checkpoint_dir = './checkpoint/'

def load_config(config_path):
	assert(os.path.exists(config_path))
	cfg = json.load(open(config_path, 'r'))
	return cfg

def build_data_loader(mode):
	"""
	Build dataloader 
	Args: 
	Return: dataloader for training, validation 
	"""
	if mode == 'train':
		train_dataset_path = ['./dataset/synthetic/dataset_200_1.json', './dataset/synthetic/dataset_200_2.json', './dataset/synthetic/dataset_200_3.json']#, './dataset/synthetic/dataset_200_n1.json', './dataset/synthetic/dataset_200_n2.json', './dataset/synthetic/dataset_200_n3.json', './dataset/synthetic/dataset_200_n4.json', './dataset/synthetic/dataset_200_n5.json', './dataset/synthetic/dataset_200_n6.json', './dataset/synthetic/dataset_200_n7.json']
		val_dataset_path = ['./dataset/synthetic/dataset_200_4.json']

		train_dataset = FlcDataset(train_dataset_path, split='train')
		train_loader = DataLoader(
			train_dataset, batch_size=args.batch_size, shuffle= True,
			num_workers=args.workers, pin_memory=False)

		val_dataset = FlcDataset(val_dataset_path, split='val')
		val_loader = DataLoader(
			val_dataset, batch_size=1, shuffle=False,
			num_workers=0, pin_memory=False)   

		print('Got {} training examples'.format(len(train_loader.dataset)))
		print('Got {} validation examples'.format(len(val_loader.dataset)))

		return train_loader, val_loader

	elif mode == 'predict':
		test_dataset_path = ['./dataset/synthetic/dataset_200_5.json']#['./dataset/synthetic/dataset_200_6.json']

		test_dataset = FlcDataset(test_dataset_path, split='test')
		test_loader = DataLoader(
			test_dataset, batch_size=1, shuffle=False,
			num_workers=0, pin_memory=False)  

		print('Got {} test examples'.format(len(test_loader.dataset)))   

		return test_loader

	
def train(cfg, model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it):

	model.train()

	for batch_count, (A, label, charge_weight, f_num, index) in enumerate(train_loader):

		input = torch.ones([args.batch_size, cfg['data']['total_nodes'],
							cfg['network']['input_feature']], dtype=torch.float64)
		input = charge_weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])
		#input[:,:,0] = charge_weight
		#input[:,:,1] = mask

		#cuda
		input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
		label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)

		optimizer.zero_grad()

		output = model(input, A)

		#mask = mask.to(device, dtype=torch.float)
		
		#loss = F.binary_cross_entropy(output, label, weight=mask)
		loss = F.binary_cross_entropy(output, label)

		loss.backward()
		optimizer.step()

		per_loss = loss.item() / args.batch_size

		tb_writer.add_scalar('train/overall_loss', per_loss, total_tb_it)
		total_tb_it += 1

		if batch_count%10 == 0:
			print('Epoch [%d/%d] Loss: %.6f' %(epoch, args.epochs, per_loss))

	return total_tb_it
		

def validate(cfg, model, val_loader, device, tb_writer, total_tb_it):
	
	model.eval()

	tb_loss = 0

	with torch.no_grad():

		for batch_count, (A, label, charge_weight, f_num, index) in tqdm(enumerate(val_loader)):

			input = torch.ones([1, cfg['data']['total_nodes'],
							cfg['network']['input_feature']], dtype=torch.float64)
			input = charge_weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])
			#input[:,:,0] = charge_weight
			#input[:,:,1] = mask

			#cuda
			input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
			label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)

			output = model(input, A)

			#mask = mask.to(device, dtype=torch.float)

			#loss = F.binary_cross_entropy(output, label, weight=mask)
			loss = F.binary_cross_entropy(output, label)

			tb_loss += loss.item()


		avg_tb_loss = tb_loss / len(val_loader.dataset)

		print('##Validate loss : %.6f' %(avg_tb_loss))

		tb_writer.add_scalar('val/overall_loss', avg_tb_loss, total_tb_it)

	return avg_tb_loss

def test(cfg, model, test_loader, device):

	model.eval()

	test_loss = 0
	hard_threshold = 0.8

	test_data = test_loader.dataset.data

	with torch.no_grad():

		for batch_count, (A, label, charge_weight, f_num, index) in enumerate(test_loader):

			input = torch.ones([1, cfg['data']['total_nodes'],
							cfg['network']['input_feature']], dtype=torch.float64)
			input = charge_weight.expand(-1, cfg['data']['total_nodes'], cfg['network']['input_feature'])

			#cuda
			input, A = input.to(device, dtype=torch.float), A.to(device, dtype=torch.float)
			label, charge_weight = label.to(device, dtype=torch.float), charge_weight.to(device, dtype=torch.float)

			output = model(input, A)

			#loss = F.binary_cross_entropy(output, label, weight=charge_weight)
			loss = F.binary_cross_entropy(output, label)

			test_loss += loss.item()

			test_data[index]['possibility_x'] = list(zip(test_data[index]['x'], torch.squeeze(output, 0).cpu().numpy()[:f_num, 0].tolist()))

			thresholed_output = torch.squeeze(output, 0).cpu().numpy()[:f_num, 0]
			thresholed_output[thresholed_output > hard_threshold] = 1
			thresholed_output[thresholed_output <= hard_threshold] = 0

			test_data[index]['predict_x'] = thresholed_output.tolist()

		avg_test_loss = test_loss / len(test_loader.dataset)

		print('##Test loss : %.6f' %(avg_test_loss))

	s_data = {'cfg':test_loader.dataset.cfg, 'data':test_data, 'loss':avg_test_loss}
	file_name = test_loader.dataset.dataset_path[0].split('.json')[0] + '_output.json'
	with open(file_name, 'w') as fp:
		fp.write(json.dumps(s_data, indent=3))



def main():
	global args
	args = parser.parse_args()
	cfg = load_config(args.config)

	device = torch.device("cuda:0" if not args.no_cuda else "cpu")
	
	if args.mode == 'train':

		exp_name = 'gcn' + '_wc1_small_bs_4_lr_5e-4'

		tb_writer = SummaryWriter(tb_log_dir + exp_name)

		train_loader, val_loader = build_data_loader(args.mode)

		model = GCN(cfg).to(device)

		optimizer = optim.Adam(model.parameters(), args.learning_rate, (0.9, 0.999), eps=1e-08)
		#optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=0.9)

		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

		total_tb_it = 0
		best_val_loss = 1000

		for epoch in range(args.epochs):
			scheduler.step(epoch)

			total_tb_it = train(cfg, model, train_loader, device, optimizer, epoch, tb_writer, total_tb_it)
			val_loss = validate(cfg, model, val_loader, device, tb_writer, total_tb_it)

			if val_loss <= best_val_loss:
				best_val_loss = val_loss
				name = checkpoint_dir + exp_name +'_model.pkl'
				state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
				torch.save(state, name)
			
		tb_writer.close()
		
	elif args.mode == 'predict':

		print('Load data...')
		test_loader = build_data_loader(args.mode)

		print('Start predicting...')

		model = GCN(cfg).to(device)

		model.load_state_dict(torch.load(args.resume_file)['model_state'])

		#summary(model, [(200, 1), (200, 200)])

		test(cfg, model, test_loader, device)

	else:
		raise Exception("Unrecognized mode.")
	
	
	
if __name__ == '__main__':
	main()

