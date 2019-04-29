import math

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GCN(nn.Module):
	def __init__(self, cfg, nclass=1, dropout=0.5):
		super().__init__()

		#self.l1 = nn.Linear(1, 4)
		#self.l2 = nn.Linear(4, 16)
		#self.l3 = nn.Linear(16, 32)


		self.gc1 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['input_feature'], cfg['network']['hidden_layer1'])
		self.gc2 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['hidden_layer1'], cfg['network']['hidden_layer2'])
		self.gc3 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['hidden_layer2'], cfg['network']['hidden_layer5'])
		#self.gc4 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['hidden_layer3'], cfg['network']['hidden_layer4'])
		#self.gc5 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['hidden_layer4'], cfg['network']['hidden_layer5'])
		self.gc6 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['hidden_layer5'], cfg['network']['hidden_layer6'])
		self.gc7 = GraphConvolution(cfg['data']['total_nodes'], cfg['network']['hidden_layer6'], nclass)

		#self.dropout = dropout

	def forward(self, x, adj):
		#x = F.relu(self.l1(x))
		#x = F.relu(self.l2(x))
		#x = F.relu(self.l3(x))

		x = F.relu(self.gc1(x, adj))

		x = F.relu(self.gc2(x, adj))

		x = F.relu(self.gc3(x, adj))

		#x = F.relu(self.gc4(x, adj))

		#x = F.relu(self.gc5(x, adj))

		x = F.relu(self.gc6(x, adj))

		#x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc7(x, adj)
		return torch.sigmoid(x)


class GraphConvolution(nn.Module):

	def __init__(self, total_node, in_features, out_features, bias=True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(total_node, out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.matmul(input, self.weight)
		output = torch.matmul(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output
