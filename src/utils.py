import pickle
import random
import numpy as np
import scipy.sparse
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score
from collections import defaultdict
from datetime import datetime
import os
import torch
from typing import Tuple
from collections import defaultdict
from typing import Optional, Callable
from scipy.sparse import csc_matrix

"""
	Utility functions to handle data and evaluate model.
"""

class log:
	def __init__(self, model_name=None, data_name=None):
		self.time_step = str(datetime.now())
		self.save_dir_path = f".\\data\\Save_model({model_name})"
		self.log_dir_path = f".\\log({data_name}, {model_name})"
		self.log_file_name = f"({model_name})" + str(self.time_step).split(":")[-1] + ".log"
		self.train_log_path = os.path.join(self.log_dir_path, "train", self.log_file_name)
		self.valid_log_path = os.path.join(self.log_dir_path, "valid", self.log_file_name)
		self.test_log_path = os.path.join(self.log_dir_path, "test", self.log_file_name)
		self.multi_run_log_path = os.path.join(self.log_dir_path, "multiple-run", self.log_file_name)
		os.makedirs(os.path.join(self.log_dir_path, "train"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "valid"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "test"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "multiple-run"), exist_ok=True)

	def write_train_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.train_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def write_valid_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.valid_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def write_test_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.test_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def multi_run_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.multi_run_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()


def load_data(data, prefix='data/', graph_id=None):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""
	# yml 파일에 설정된 데이터셋(data_name)에 따라 label, feature, relation을 불러옴. 
	if data == 'yelp':
		prefix = "./data/pyg/YelpChi/processed/"
		data_file = torch.load(prefix + "YelpChi_data.pt")[0]
		labels = np.array(data_file['review']['y'])
		feat_data = np.array(data_file['review']['x'])

		# load the preprocessed adj_lists
		with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
		relation_list = [relation1, relation2, relation3]

	elif data == 'amazon':
		prefix = "./data/pyg/AmazonFraud/processed/"
		data_file = torch.load(prefix + "AmazonFraud_data.pt")[0]

		labels = np.array(data_file['user']['y'])
		feat_data = np.array(data_file['user']['x'])

		# load the preprocessed adj_lists
		with open(prefix + 'amazon_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amazon_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amazon_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amazon_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
		relation_list = [relation1, relation2, relation3]
  
	elif data == 'amazon_new':
		prefix = "./data/pyg/AmazonFraud/processed/"
		data_file = torch.load(prefix + "AmazonFraud_new_data.pt")

		labels = np.array(data_file['user'].y)
		feat_data = np.array(data_file['user'].x)

		# load the preprocessed adj_lists
		with open(prefix + 'amazon_new_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amazon_new_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amazon_new_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amazon_new_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
		relation_list = [relation1, relation2, relation3]
 
 
	elif data == 'tfinance':
		prefix = "./data/pyg/TFinance/processed/"
		data_file = torch.load(prefix + "tfinance_data.pt")[0]

		labels = np.array(data_file['y'])
		feat_data = np.array(data_file['x'])

		# load the preprocessed adj_lists
		with open(prefix + 'tfinance_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'tfinance_homo_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		relation_list = [relation1]

	elif data == 'elliptic':
		prefix = "./data/pyg/Elliptic/processed/"
		data_file = torch.load(prefix + "elliptic_data.pt")[0]

		labels = np.array(data_file['y'])
		feat_data = np.array(data_file['x'])

		# load the preprocessed adj_lists
		with open(prefix + 'elliptic_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'elliptic_homo_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		relation_list = [relation1]

	elif data == 'weibo':
		prefix = "./data/pyg/Weibo/processed/"
		data_file = torch.load(prefix + "weibo.pt")[0]

		labels = np.array(data_file['y'])
		feat_data = np.array(data_file['x'])

		# load the preprocessed adj_lists
		with open(prefix + 'weibo_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'weibo_homo_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		relation_list = [relation1]	

	elif data == 'kdk':
		postfix = "(CSC).npz"

		graph_num = str(graph_id).zfill(3)
		feature_path = os.path.join(prefix, "attributes", graph_num + "_node_feature" + postfix)
		label_path = os.path.join(prefix, "labels", graph_num + "_label.npy")

		labels = np.load(label_path).flatten()
		feat_data = scipy.sparse.load_npz(feature_path).astype(float).todense().A

		network_dir_path_hetero = os.path.join(prefix, "G0_Hetero")
		network_type_list = ["_c_acc_c_network", "_c_clcare_c_network", "_c_fp_c_network",
						"_c_hsdrcare_c_network","_c_insr_c_network"]
		network_path_list = [os.path.join(network_dir_path_hetero, graph_num + network_type_list[i] + postfix) for i in range(len(network_type_list))]
		relation_list = [scipy.sparse.load_npz(network_path_list[i]) for i in range(len(network_path_list))]
		for i, relation in enumerate(relation_list):
			relation_list[i] = sparse_to_adjlist_for_train(relation + sp.eye(relation.shape[0]))

		network_dir_path_homo = os.path.join(prefix, "G0_Homo")
		homo_network_path = os.path.join(network_dir_path_homo, graph_num + "_G0_Homo_network" + postfix)
		homo = scipy.sparse.load_npz(homo_network_path)
		homo = sparse_to_adjlist_for_train(homo + sp.eye(homo.shape[0]))


	return homo, relation_list, feat_data, labels


def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_to_adjlist(sp_matrix, filename): # CSC 포멧의 인접행렬을 adjlist 형태로 변환하는 함수로 data_process.py에서 사용되는 함수.
	"""
	Transfer sparse matrix to adjacency list
	:param sp_matrix: the sparse matrix
	:param filename: the filename of adjlist
	"""
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0]) 
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero() # 해당 함수는 non-zero value를 갖는 인덱스 (row, col)을 리스트 퓨플로 반환한다.
	for index, node in enumerate(edges[0]): 
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node) # symmetric relation을 커버하기 위한 코드.
	with open(filename, 'wb') as file:
		pickle.dump(adj_lists, file)
	file.close()

def sparse_to_adjlist_for_train(sp_matrix): # CSC 포멧의 인접행렬을 adjlist 형태로 변환하는 함수로 학습 과정에서 사용한다.
	# add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0]) 
	# create adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero() # 해당 함수는 non-zero value를 갖는 인덱스 (row, col)을 리스트 퓨플로 반환한다.
	for index, node in enumerate(edges[0]): 
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node) # symmetric relation을 커버하기 위한 코드.

	return adj_lists

def pos_neg_split(nodes, labels): # label을 기준으로 positive와 netgative sample의 노드 번호를 반환하는 함수.
	"""
	Find positive and negative nodes given a list of nodes and their labels
	:param nodes: a list of nodes
	:param labels: a list of node labels
	:returns: the spited positive and negative nodes
	"""
	pos_nodes = []
	neg_nodes = cp.deepcopy(nodes)
	aux_nodes = cp.deepcopy(nodes)
	for idx, label in enumerate(labels):
		if label == 1:
			pos_nodes.append(aux_nodes[idx])
			neg_nodes.remove(aux_nodes[idx])

	return pos_nodes, neg_nodes # Positive와 netgative sample의 노드 인덱스를 반환 (전체 데이터셋으로부터의 인덱스가 기준이 됨.).


def pick_step(idx_train, y_train, adj_list, size): # 논문에서 제안한 label balance sampler에 해당하는 함수.
	degree_train = [len(adj_list[node]) for node in idx_train] # adj list (homo)를 통해 train 노드들의 차수를 리스트로 반환한다 (self loop 포함됨).
	lf_train = (y_train.sum()-len(y_train))*y_train + len(y_train) 
	smp_prob = np.array(degree_train) / lf_train # Sampling probability를 (P(v))를 구한다 (해당 노드의 label의 수와 차수의 비율로 설정하여 imbalance problem 완화함.).
	return random.choices(idx_train, weights=smp_prob, k=size)

def test(test_nodes: np.array, labels: np.array, model, batch_size: int, 
	result=None, epoch: Optional[int]=None, epoch_best: Optional[int]=None,
	flag: Optional[str]=None, print_line: Optional[bool]=True) -> Tuple[float, float, float, float]:
        
	"""
	Test the performance of the model (F1, Accuracy, Recall, AUC-ROC).
	
	Determine the best threshold with repect to F1 score.
	Calculate the Accuracy, Recall with the threshold.
	
	Prediction_list comprises the index of max value for each row.
	Conficene_list comprises the value of argmax for each row.
	"""

	# output_list[0]: prediction_list, Prediction of the model.
	# output_list[1]: confidence_list, Confidence for the max prediction.
	# output_list[2]: anomaly_confidence_list, Confidence of the predicted anomalies.
	output_list = [[], [], []]
	num_batch = int(len(test_nodes) / batch_size) + 1 # Performance test with batch process.
	for i in range(num_batch):
		start = i * batch_size
		end = min((i + 1) * batch_size, len(test_nodes))
		batch_nodes = test_nodes[start:end]
		batch_labels = labels[start:end]

		output = model.to_prob(batch_nodes, batch_labels, train_flag=False)[0].data.cpu().numpy() # Casting the output tensor to numpy array. 
		prediction = output.argmax(axis=1) # The argmax value represents the prediction(index) of the model. 
		confidence = output.max(axis=1) # Tha max value represents the conficende for each the prediction(index).
		anomaly_confidence = output[:,1]
		
		output_list[0].extend(prediction.tolist()) # Combining the predictions into the list
		output_list[1].extend(confidence.tolist())
		output_list[2].extend(anomaly_confidence.tolist())
	
	output_list = np.array(output_list) # Casting the list to numpy array. 
	
	accuracy = accuracy_score(labels, output_list[0])
	f1 = f1_score(labels, output_list[0])
	f1_macro = f1_score(labels, output_list[0], average='macro')
	precision = precision_score(labels, output_list[0], zero_division=0)
	precision_macro = precision_score(labels, output_list[0], zero_division=0, average='macro')
	recall = recall_score(labels, output_list[0])
	recall_macro = recall_score(labels, output_list[0], average='macro')
	auc = roc_auc_score(labels, output_list[2])
	
	line= f"- F1: {f1:.4f}\t- Recall: {recall:.4f}\t- Precision: {precision:.4f}\t- Accuracy: {accuracy:.4f}\t- AUC-ROC: {auc:.4f}\t- F1-macro: {f1_macro:.4f}\t- Recall-macro: {recall_macro:.4f}\t- AP: {precision_macro:.4f}\t\n"

	if not (result is None):
		if flag=="val":
			result.write_val_log(epoch, epoch_best, accuracy, f1, f1_macro, precision, precision_macro, recall, recall_macro, auc, line, print_line)
		elif flag=="test":
			result.write_test_log(epoch_best, accuracy, f1, f1_macro, precision, precision_macro, recall, recall_macro, auc, line, print_line)

	return auc, recall, f1_macro, precision


# def test_sage(test_cases, labels, model, batch_size, ckp, thres=0.5, flag=None):
# 	"""
# 	Test the performance of GraphSAGE
# 	:param test_cases: a list of testing node
# 	:param labels: a list of testing node labels
# 	:param model: the GNN model
# 	:param batch_size: number nodes in a batch
# 	"""

# 	test_batch_num = int(len(test_cases) / batch_size) + 1
# 	f1_gnn = 0.0
# 	acc_gnn = 0.0
# 	recall_gnn = 0.0
# 	gnn_list = []
# 	for iteration in range(test_batch_num):
# 		i_start = iteration * batch_size
# 		i_end = min((iteration + 1) * batch_size, len(test_cases))
# 		batch_nodes = test_cases[i_start:i_end]
# 		batch_label = labels[i_start:i_end]

# 		gnn_prob = model.to_prob(batch_nodes)

# 		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
# 		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
# 		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

# 		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

# 	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
# 	ap_gnn = average_precision_score(labels, np.array(gnn_list))


# 	line1= f"GNN F1: {f1_gnn / test_batch_num:.4f}\tGNN Accuracy: {acc_gnn / test_batch_num:.4f}"+\
# 	   f"\tGNN Recall: {recall_gnn / test_batch_num:.4f}\tGNN auc: {auc_gnn:.4f}\tGNN ap: {ap_gnn:.4f}"

# 	if flag=="val":
# 		ckp.write_valid_log("Validation: "+ line1)
# 	elif flag=="test":
# 		ckp.write_test_log("Test: "+ line1)

# 	return auc_gnn, (recall_gnn / test_batch_num), (f1_gnn / test_batch_num)

# def test_pcgnn(test_cases, labels, model, batch_size, ckp, thres=0.5, flag=None):
# 	"""
# 	Test the performance of CARE-GNN and its variants
# 	:param test_cases: a list of testing node
# 	:param labels: a list of testing node labels
# 	:param model: the GNN model
# 	:param batch_size: number nodes in a batch
# 	:returns: the AUC and Recall of GNN and Simi modules
# 	"""

# 	test_batch_num = int(len(test_cases) / batch_size) + 1
# 	f1_gnn = 0.0
# 	acc_gnn = 0.0
# 	recall_gnn = 0.0

# 	f1_label1 = 0.0
# 	acc_label1 = 0.00
# 	recall_label1 = 0.0
# 	gnn_list = []
# 	label_list1 = []

# 	for iteration in range(test_batch_num):
# 		i_start = iteration * batch_size
# 		i_end = min((iteration + 1) * batch_size, len(test_cases))
# 		batch_nodes = test_cases[i_start:i_end]
# 		batch_label = labels[i_start:i_end]


# 		# 학습된 CARE-GNN 모델을 통해 반환되는 GNN score와 label-aware score를 통해 성능 평가를 수행한다.
# 		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

# 		f1_gnn += f1_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
# 		acc_gnn += accuracy_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1))
# 		recall_gnn += recall_score(batch_label, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")

# 		f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
# 		acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
# 		recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

# 		gnn_list.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())
# 		label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

# 	auc_gnn = roc_auc_score(labels, np.array(gnn_list))
# 	ap_gnn = average_precision_score(labels, np.array(gnn_list))
# 	auc_label1 = roc_auc_score(labels, np.array(label_list1))
# 	ap_label1 = average_precision_score(labels, np.array(label_list1))
# # 	line1= f"GNN F1: {f1_gnn / test_batch_num:.4f}\tGNN Accuracy: {acc_gnn / test_batch_num:.4f}"+\
# #        f"\tGNN Recall: {recall_gnn / test_batch_num:.4f}\tGNN auc: {auc_gnn:.4f}\tGNN ap: {ap_gnn:.4f}"
# # 	line2 = f"Label1 F1: {f1_label1 / test_batch_num:.4f}\tLabel1 Accuracy: {acc_label1 / test_batch_num:.4f}"+\
# #        f"\tLabel1 Recall: {recall_label1 / test_batch_num:.4f}\tLabel1 auc: {auc_label1:.4f}\tLabel1 ap: {ap_label1:.4f}"

# 	line1= f"GNN F1: {f1_gnn / test_batch_num:.4f}\tGNN Accuracy: {acc_gnn / test_batch_num:.4f}"+\
# 	   f"\tGNN Recall: {recall_gnn / test_batch_num:.4f}\tGNN AUC-ROC: {auc_gnn:.4f}\tGNN AP: {ap_gnn:.4f}"

# 	if flag=="val":
# 		ckp.write_valid_log("Validation: "+ line1)
# 		# ckp.write_valid_log("Validation: "+ line2, print_line=False)
# 	elif flag=="test":
# 		ckp.write_test_log("Test: "+ line1)
# 		# ckp.write_test_log("Test: "+ line2, print_line=False)

# 	return auc_gnn, (recall_gnn / test_batch_num), (f1_gnn / test_batch_num)

def prob2pred(y_prob, thres=0.5):
	"""
	Convert probability to predicted results according to given threshold
	:param y_prob: numpy array of probability in [0, 1]
	:param thres: binary classification threshold, default 0.5
	:returns: the predicted result with the same shape as y_prob
	"""
	y_pred = np.zeros_like(y_prob, dtype=np.int32)
	y_pred[y_prob >= thres] = 1
	y_pred[y_prob < thres] = 0
	return y_pred


def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5

def create_dir(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError:
        print("Error: Failed to create the directory.")

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)