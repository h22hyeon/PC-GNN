import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from operator import itemgetter
import math

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists: # 각 relation을 통해 연결되는 이웃 노드들의 set으로 구성된 list가 생성됨.
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		"""
		Relation에 따른 노드 set을 구하는 부분이 하드코딩 되어있음.
		따라서 relation type이 다른 데이터셋을 학습할 경우 일부 코드 수정이 필요할 것으로 보임 (파서로 자동화 혹은 하드코딩으로 해결).
		"""
		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]), # 배치에 포함된 노드와 그 이웃 노드들의 set이 생성됨.
								 set.union(*to_neighs[2], set(nodes)))

		# calculate label-aware scores
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes))) # unique node들에 대한 feature를 슬라이싱하여 batch_feature로 정의함.
			pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos))) # 그 중에서 positive sample의 feature를 슬라이싱하여 pos_features로 정의함.
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.LongTensor(list(self.train_pos)))
		batch_scores = self.label_clf(batch_features) # batch_features를 latent space로 투영함.
		pos_scores = self.label_clf(pos_features) # pos_features를 latent space로 투영함.

		# 배치를 구성하는 노드의 ID(original graph의 index -> key)와 unique_nodes에서의 인덱스(-> value)를 매핑하는 딕셔너리를 정의함.
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :] # mapping 딕셔너리를 통해 배치 노드의 label-aware score를 구한다 (슬라이싱).

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]] # 각 relation을 통해 연결되는 이웃 노드들을 r*_list로 정의한다 (set을 list로 변환함.).
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list] # 각 relation마다
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list] # Batch에 존재하는 [개별 노드와 그 이웃에 대한 label-aware score]를 구한다 (슬라이싱). 
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		""""
		이 부분에서 threshold에 대한 부분은 하드코딩 되어 있음. 
		따라서 모든 relation에 대하여 각 노드는 이웃의 유사도가 상위 50%에 해당하는 노드들로부터 message를 받게됨. 
		"""
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list] # 각 relation마다
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list] # Batch에 존재하는 개별 노드가 몇 개의 이웃을 통해 aggregation 할 것인지 결정한다. 
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		# r1_feats은 배치의 각 노드에 대한 representation을 의미한다.
		# r1_scores은 각 타겟 노드에 대한 선택된 이웃 노드들의 score diff를 의미한다 (이용하려면 매핑되는 노드의 인덱스가 필요함. 현재는 사용 X).
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, labels, r1_list, center_scores, r1_scores, pos_scores, r1_sample_num_list, train_flag)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, labels, r2_list, center_scores, r2_scores, pos_scores, r2_sample_num_list, train_flag)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, labels, r3_list, center_scores, r3_scores, pos_scores, r3_sample_num_list, train_flag)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# concat the intra-aggregated embeddings from each relation
		# Eq. (9) in the paper
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)

		"""
		각 relation의 embedding을 통합하기 위한 weight를 학습해야 한다!
		"""
		combined = F.relu(cat_feats.mm(self.weight).t()) # intra-aggregated embeddings이다!

		return combined, center_scores # 통합된 embedding과 배치의 각 노드에 대한 label-aware score(어디에 쓰이는지..? -> latent space로 투영하는 가중치의 학습을 위해)를 반환한다.


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param rho: the ratio of the oversample neighbors for the minority class
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.rho = rho
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list, train_flag):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param | nodes: list of nodes in a batch
		:param | to_neighs_list: neighbor node id list for each batch node in one relation
		:param | batch_scores: the label-aware scores of batch nodes
		:param | neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param | pos_scores: the label-aware scores for the minority positive nodes
		:param | sample_list: the number of neighbors kept for each batch node in one relation
		:param | train_flag: indicates whether in training or testing mode

		# shape 정리
		# nodes: [B,] -> node_idx
		# batch_labels : [B,] -> 0, 1
		# to_neighs_list : [B,] -> [neighbor1_idx, neighbor2_idx, ...]
		# batch_scores : [B, 2] -> scores
		# neigh_scores: [B,] -> [scores of neighbor1, scores of neighbor2, ...]
		# pos_scores : [# of pos, 2] -> 
		# sample_list: [B,] -> threshold * k

		:return | to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return | samp_scores: the average neighbor distances for each relation after filtering
		"""

		# filer neighbors under given relation in the train mode
		if train_flag:
			# choose_step_neighs는 배치에 존재하는 개별 노드에 대한 선택된 이웃 노드와 그들의 score diff를 반환한다.
			samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
		else:
			samp_neighs, samp_scores = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)
		
		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs)) # 동일한 노드들이 존재할 수 있으므로 중복되는 노드를 제거한다.
		# 이웃 노드들에 대해 개별적으로 접근해야 하므로 노드 인덱스에 대한 딕셔너리를 정의한다. 
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		# 배치의 각 노드에 대한 이웃 노드들에 차례로 접근하여 인덱스를 부여한다.
		# (이때 모든 배치의 노드에 대해 하나의 리스트로 결과가 생성한다. -> column_indices)
		# 타겟 노드의 이웃 노드의 인덱스를 하나의 리스트로 통합하여 column_indices로 정의한다.
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		# column_indices로에서 이웃 노드의 인덱스가 몇 번째 타겟 노드의 것인지를 지정하기 위한 row_indices를 생성한다.
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		# mask의 row는 타겟 노드의 인덱스를 의미하고, column은 이웃 노드의 인덱스를 의미한다.
		# 배치 단위에서의 adj라고 보면 될 것 같다. 
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		# row-sum은 이웃 노드의 수!
		num_neigh = mask.sum(1, keepdim=True)
		# mean aggregator!
		mask = mask.div(num_neigh)
		if self.cuda:
			# 배치의 노드와 그 이웃 노드들의 노드 피처!
			self_feats = self.features(torch.LongTensor(nodes).cuda())
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		
		# agggregation 과정을 수행한다!
		agg_feats = mask.mm(embed_matrix)  # single relation aggregator
		cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # concat with last layer
		"""
		# 단일 relation의 embedding을 생성하기 위한 weight를 학습해야 한다!
		"""
		to_feats = F.relu(cat_feats.mm(self.weight))
		return to_feats, samp_scores # node representation과 각 타겟 노드에 대한 선택된 이웃 노드들의 score diff를 반환한다.


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
    """
    Choose step for neighborhood sampling
    :param | center_scores: the label-aware scores of batch nodes
    :param | center_labels: the label of batch nodes
    :param | neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param | neighs_list: neighbor node id list for each batch node in one relation
	:param | minor_scores: the label-aware scores for nodes of minority class in one relation
    :param | minor_list: minority node id list for each batch node in one relation
    :param | sample_list: the number of neighbors kept for each batch node in one relation
	:param  |sample_rate: the ratio of the oversample neighbors for the minority class
	choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
    """
    samp_neighs = []
    samp_score_diff = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0] # 인덱스에 해당하는 타겟 노드의 label-aware score [0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1) # 타겟 노드의 이웃 노드들의 label-aware score [0]
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1) # 이웃 노드의 수 만큼 타겟 노드의 label-aware score를 확장한다.
        neighs_indices = neighs_list[idx] # 이웃 노드의 인덱스를 neighs_indices로 저장한다.
        num_sample = sample_list[idx] # 타겟 노드에 대하여 message를 이용할 이웃 노드의 수를 num_sample로 저장한다.

        # compute the L1-distance of batch nodes and their neighbors
		# 각 타겟 노드에 대하여 이웃 노드와의 labe-aware score를 게산하여 score_diff로 정의 한다.
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze() # 이웃 노드들과 score diff를 계산하고, 이를 정렬한다. 
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist() # 

        # top-p sampling according to distance ranking
        if len(neigh_scores[idx]) > num_sample + 1:
			# 선택된 이웃 노드들의 인덱스를 selected_neighs로 정의한다.
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
	    	# 선택된 이웃 노드들과 타겟 노드와의 score 차이를  selected_score_diff로 정의한다.
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
			# 타겟 노드의 이웃 노드가 1개 혹은 singleton 노드일 경우
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

		# 타겟 노드가 fraud일 경우
        if center_labels[idx] == 1:
			# over sampling의 비율은 하이퍼 파라미터이다.
            num_oversample = int(num_sample * sample_rate)
	    	# Positive sample의 label-aware score와 비교하기 위해 타겟 노드의 label-aware score를 확장한다.
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
	    	# 타겟 노드와 positive sample의 score diff를 계산한다.
            score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
	    	# 마찬가지로 위 score diff와 인덱스를 정렬한다.
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
	    	# over sampling된 positive sample을 이웃 노드로 추가한다.
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

		# 노드들의 선택된 neighbor와 그들의 score diff를 저장한다. 
        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff # 배치에 존재하는 각 노드들의 이웃의 인덱스와 그들의 score diff를 반환한다.


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
	"""
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
		selected_indices = sorted_indices.tolist()

		# top-p sampling according to distance ranking and thresholds
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores
