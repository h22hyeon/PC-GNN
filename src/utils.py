import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from collections import defaultdict
from datetime import datetime
import os


"""
	Utility functions to handle data and evaluate model.
"""

class log:
	def __init__(self):
		self.log_dir_path = "./log"
		self.log_file_name = datetime.now().strftime("%Y-%m-%d %H:%M") + ".log"
		self.train_log_path = os.path.join(self.log_dir_path, "train", self.log_file_name)
		self.test_log_path = os.path.join(self.log_dir_path, "test", self.log_file_name)
		self.multi_run_log_path = os.path.join(self.log_dir_path, "multi-run(total)", self.log_file_name)
		os.makedir(os.path.join(self.log_dir_path, "train"), exist_ok=True)
		os.makedir(os.path.join(self.log_dir_path, "test"), exist_ok=True)
		os.makedir(os.path.join(self.log_dir_path, "multiple-run"), exist_ok=True)

	def write_train_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.train_log_path, 'a')
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



def load_data(data, prefix='data/'):
	"""
	Load graph, feature, and label given dataset name
	:returns: home and single-relation graphs, feature, label
	"""
	# yml 파일에 설정된 데이터셋(data_name)에 따라 label, feature, relation을 불러옴. 
	if data == 'yelp':
		data_file = loadmat(prefix + 'YelpChi.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
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
		data_file = loadmat(prefix + 'Amazon.mat')
		labels = data_file['label'].flatten()
		feat_data = data_file['features'].todense().A
		# load the preprocessed adj_lists
		with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
			homo = pickle.load(file)
		file.close()
		with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
			relation1 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
			relation2 = pickle.load(file)
		file.close()
		with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
			relation3 = pickle.load(file)
		file.close()
		relation_list = [relation1, relation2, relation3]

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


def test_sage(test_cases, labels, model, batch_size, thres=0.5):
	"""
	Test the performance of GraphSAGE
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	gnn_pred_list = []
	gnn_prob_list = []
	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob = model.to_prob(batch_nodes)

		gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
		gnn_pred = prob2pred(gnn_prob_arr, thres)
		
		gnn_pred_list.extend(gnn_pred.tolist())
		gnn_prob_list.extend(gnn_prob_arr.tolist())
	
	auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
	f1_binary_1_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=1, average='binary')
	f1_binary_0_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=0, average='binary')
	f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
	f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
	conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
	tn, fp, fn, tp = conf_gnn.ravel()
	gmean_gnn = conf_gmean(conf_gnn)

	print(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"+
			f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
	print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
	return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn

	

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


def test_pcgnn(test_cases, labels, model, batch_size, thres=0.5):
	"""
	Test the performance of PC-GNN and its variants
	:param test_cases: a list of testing node
	:param labels: a list of testing node labels
	:param model: the GNN model
	:param batch_size: number nodes in a batch
	:returns: the AUC and Recall of GNN and Simi modules
	"""

	test_batch_num = int(len(test_cases) / batch_size) + 1
	f1_gnn = 0.0
	acc_gnn = 0.0
	recall_gnn = 0.0
	f1_label1 = 0.0
	acc_label1 = 0.00
	recall_label1 = 0.0
	gnn_pred_list = []
	gnn_prob_list = []
	label_list1 = []

	for iteration in range(test_batch_num):
		i_start = iteration * batch_size
		i_end = min((iteration + 1) * batch_size, len(test_cases))
		batch_nodes = test_cases[i_start:i_end]
		batch_label = labels[i_start:i_end]
		gnn_prob, label_prob1 = model.to_prob(batch_nodes, batch_label, train_flag=False)

		gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]
		gnn_pred = prob2pred(gnn_prob_arr, thres)

		f1_label1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
		acc_label1 += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
		recall_label1 += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

		gnn_pred_list.extend(gnn_pred.tolist())
		gnn_prob_list.extend(gnn_prob_arr.tolist())
		label_list1.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())

	auc_gnn = roc_auc_score(labels, np.array(gnn_prob_list))
	ap_gnn = average_precision_score(labels, np.array(gnn_prob_list))
	auc_label1 = roc_auc_score(labels, np.array(label_list1))
	ap_label1 = average_precision_score(labels, np.array(label_list1))

	f1_binary_1_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=1, average='binary')
	f1_binary_0_gnn = f1_score(labels, np.array(gnn_pred_list), pos_label=0, average='binary')
	f1_micro_gnn = f1_score(labels, np.array(gnn_pred_list), average='micro')
	f1_macro_gnn = f1_score(labels, np.array(gnn_pred_list), average='macro')
	conf_gnn = confusion_matrix(labels, np.array(gnn_pred_list))
	tn, fp, fn, tp = conf_gnn.ravel()
	gmean_gnn = conf_gmean(conf_gnn)

	print(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}"+
			f"\tF1-macro: {f1_macro_gnn:.4f}\tG-Mean: {gmean_gnn:.4f}\tAUC: {auc_gnn:.4f}")
	print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
	print(f"Label1 F1: {f1_label1 / test_batch_num:.4f}\tAccuracy: {acc_label1 / test_batch_num:.4f}"+
	      f"\tRecall: {recall_label1 / test_batch_num:.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}")

	return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn, gmean_gnn

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5