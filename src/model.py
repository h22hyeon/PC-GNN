import torch
import torch.nn as nn
from torch.nn import init


"""
	PC-GNN Model
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class PCALayer(nn.Module):
	"""
	One Pick-Choose-Aggregate layer
	"""

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the PC-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(PCALayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	def forward(self, nodes, labels, train_flag=True):
		# InterAgg 레이어는 최종적으로 노드의 inter-agrregated embedding과 label-aware score를 반환한다.
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)
		# 가중치를 통해 최종 embedding으로 투영하여 score를 계산한다.
		scores = self.weight.mm(embeds1)
		return scores.t(), label_scores # 노드에 대한 fraud score (GNN score)와 label-aware score를 반환한다.

	def to_prob(self, nodes, labels, train_flag=True):
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)
		return gnn_scores, label_scores

	def loss(self, nodes, labels, train_flag=True):
		# PCALayer는 최종적으로 GNN score와 label-aware score를 반환한다.
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
		# Simi loss, Eq. (7) in the paper
		"""
		Choose 부분에서 타겟 노드와 이웃 노드의 유사도를 측정하기 위한 투영 행렬을 학습하기 위한 loss이다. 
		"""
		label_loss = self.xent(label_scores, labels.squeeze())
		# GNN loss, Eq. (10) in the paper
		"""
		Intra/Inter aggregator에 존재하는 가중치들을 학습하기 위한 loss이다.
		"""
		gnn_loss = self.xent(gnn_scores, labels.squeeze()) # 연결된 layer들과 가중치 행렬로 역전파된다.
		# the loss function of PC-GNN, Eq. (11) in the paper
		final_loss = gnn_loss + self.lambda_1 * label_loss # loss 조정.
		return final_loss
