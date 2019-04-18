import math

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass=1):
		super().__init__()

		self.gc1 = GraphConvolution(nfeat, nhid)
		self.gc2 = GraphConvolution(nhid, nclass)

	def forward(self, x, adj):
		x = F.relu(self.gc1(x, adj))
		x = self.gc2(x, adj)
		return F.sigmoid(x)


class GraphConvolution(nn.Module):

	def __init__(self, in_features, out_features, bias=True):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(in_features, out_features))
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
			return output + torch.matmul(input, self.bias)
		else:
			return output
