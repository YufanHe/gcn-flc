import os
import sys
import json

import torch
from torch.utils import data

import numpy as np
import math
from scipy.sparse import coo_matrix


class FlcDataset(data.Dataset):

	def __init__(self, dataset_path, split='train'):
		self.dataset_path = dataset_path
		self.split = split

		self.cfg, self.data = loaddata(self.dataset_path)
		self.total_nodes = self.cfg['total_nodes']

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		
		data = self.data[index]

		graph_dict = data['graph_dict']

		adj_matrix = load_adj(graph_dict, self.total_nodes)
		DAD_matrix = normalizeA(adj_matrix, self.total_nodes)

		label = np.zeros(self.total_nodes) 
		label[:len(data['x'])] = np.array(data['x'])

		charge_weight = np.zeros(self.total_nodes)
		charge_weight[:len(data['charge'])] = np.array(data['charge'])

		return DAD_matrix, label, charge_weight, index

def loaddata(file_name):
    """
    Data will be loaded from json format
    
    Args:
        file_name: absolute path for the data file
    Return:
        cfg: configuration of the generated data
        data: data generated with ground truth
    """
    assert(os.path.isfile(file_name))
    s_data = json.load(open(file_name, 'r'))
    return s_data['cfg'], s_data['data']

def load_adj(g_dict, N):

	row  = np.array(g_dict['row'])
	col  = np.array(g_dict['col'])
	data = np.array(g_dict['data'])
	A = coo_matrix((data, (row, col)), shape=(N, N)).toarray()

	return A

def normalizeA(A, N):

	A_I = A + np.identity(N)
	D = np.diagflat(np.sum(A_I, axis=0))
	D_inv = np.linalg.inv(D)
	D_inv_2 = np.sqrt(D_inv)
	DAD = np.matmul(D_inv_2, np.matmul(A_I, D_inv_2))

	#DAD[DAD < 1e-6] = 1000
	return DAD