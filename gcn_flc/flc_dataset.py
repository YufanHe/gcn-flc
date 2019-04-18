import os
import sys
import json

import torch
from torch.utils import data

import numpy as np
import math
from scipy.sparse import coo_matrix


class FlcDataset(data.Dataset):

	def __init__(self, dataset_path, N, split='train'):
		self.dataset_path = dataset_path
		self.N = N
		self.split = split

		self.data = []

		with open(self.dataset_path, 'r') as f:
			for line in f:
				self.data.append(line.rstrip('\n'))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		
		data_dic = load_data(self.data[index])

		adj_matrix = load_adj(data_dic['graph_dict'], self.N)

		label = np.zeros(self.N) 
		label[:len(data_dic['x'])] = np.array(data_dic['x'])

		charge_weight = np.zeros(self.N)
		charge_weight[:len(data_dic['charge'])] = np.array(data_dic['charge'])
		
		return adj_matrix, label, charge_weight, self.data[index]

def load_data(data_path):
	assert(os.path.exists(data_path))
	data = json.load(open(data_path, 'r'))
	return data

def load_adj(g_dict, N):

	row  = np.array(g_dict['row'])
	col  = np.array(g_dict['col'])
	data = np.array(g_dict['data'])
	A = coo_matrix((data, (row, col)), shape=(N, N)).toarray()

	return A