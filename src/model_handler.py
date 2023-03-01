import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step
from src.model import PCALayer
from src.layers import InterAgg3, InterAgg5, IntraAgg
from src.graphsage import *


"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler(object):

	def __init__(self, config, ckp):
		self.ckp = ckp
		args = argparse.Namespace(**config)
		args.cuda = not args.no_cuda and torch.cuda.is_available()

		# 인덱싱이 잘 안되는 오류 발생해서 교체함 : os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
		device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
		torch.cuda.set_device(device)

		# load graph, feature, and label
		homo, relation_list, feat_data, labels = load_data(args.data_name, prefix=args.data_dir, graph_id=args.graph_id) # KDK 데이터 셋에서 realation의 수가 달라질 수 있어 수정함.

		# train/validation/test set 분할.
		np.random.seed(args.seed)
		random.seed(args.seed)
		if args.data_name == 'yelp' or args.data_name == 'KDK':
			index = list(range(len(labels))) # Stratified sampling을 통해 데이터셋을 train/validation/test set으로 분할한다.
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
																	random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
																	random_state=2, shuffle=True)

		elif args.data_name == 'amazon':
			# 0-3304 are unlabeled nodes (Unlabeled 노드는 학습과 검증 과정에서 제외함.)
			index = list(range(3305, len(labels))) # Stratified sampling으로 train/validation/test를 구분한다.
			idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
																	train_size=args.train_ratio, random_state=2, shuffle=True)
			idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
																	test_size=args.test_ratio, random_state=2, shuffle=True)

		print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
			f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
		print(f"Classification threshold: {args.thres}")
		print(f"Feature dimension: {feat_data.shape[1]}")


		# split pos neg sets for under-sampling
		train_pos, train_neg = pos_neg_split(idx_train, y_train)

		# Feature normalization 부분 코드.
		# if args.data == 'amazon':
		feat_data = normalize(feat_data)
		# train_feats = feat_data[np.array(idx_train)]
		# scaler = StandardScaler()
		# scaler.fit(train_feats)
		# feat_data = scaler.transform(feat_data)

		# set input graph
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = homo
		else: # PC-GNN은 multi-relational graph를 이용함.
			adj_lists = relation_list

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
		
		self.ckp = ckp
		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
						'train_pos': train_pos, 'train_neg': train_neg}


	def train(self):
		args = self.args
		
		# 클래스 변수로부터 feature, label, adj 생성한다.
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']
		
		# initialize model input
		features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
		features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
		if args.cuda:
			features.cuda()

		"""
		논문의 choosse 과정 (IntraAgg layer 안에서 이루어짐.)

		# 사용할 positive과 negative 비율을 유사하게 맞춰주기 위한 코드.
		# Train positive set의 2배를 샘플링 하여 배치 구성에서 두 클래스의 비율을 유사하게 가져가려 함.
		"""
		# build one-layer models
		if args.model == 'PCGNN' and args.data_name == "KDK":
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra4 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra5 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg5(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], adj_lists, [intra1, intra2, intra3, intra4, intra5], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'PCGNN':
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg3(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
							  adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False, cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True, cuda=args.cuda)

		if args.model == 'PCGNN':
			gnn_model = PCALayer(2, inter1, args.alpha) # 앞서 정의한 inter/intra aggregator를 이용하여 PCGNN 모델을 생성함.
		elif args.model == 'SAGE':
			# the vanilla GraphSAGE model as baseline
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		if args.cuda:
			gnn_model.cuda()

		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

		dir_saver = os.path.join("/data/PC-GNN_models/", self.ckp.log_file_name)
		os.makedirs(dir_saver,exist_ok=True)
		path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
		f1_mac_best, auc_best, ep_best = 0, 0, -1

		# train the model
		for epoch in range(args.num_epochs):
			"""
			논문의 pick 과정

			# 사용할 positive과 negative 비율을 유사하게 맞춰주기 위한 코드.
			# Train positive set의 2배를 샘플링 하여 배치 구성에서 두 클래스의 비율을 유사하게 가져가려 함.
			"""
			sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos'])*2) 
			random.shuffle(sampled_idx_train)
			num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

			loss = 0.0
			epoch_time = 0
			# mini-batch training
			"""
			자세한 학습 알고리즘은 layers.py를 통해 확인할 수 있다. 
			"""
			for batch in range(num_batches):
				start_time = time.time()
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))

				batch_nodes = sampled_idx_train[i_start:i_end]
				batch_label = self.dataset['labels'][np.array(batch_nodes)]
				optimizer.zero_grad()
				if args.cuda:
					loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
				else:
					loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
				loss.backward()
				optimizer.step()
				end_time = time.time()
				epoch_time += end_time - start_time
				loss += loss.item()
			
			train_line = f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s'
			self.ckp.write_train_log(train_line)

			"""
			Test 과정에 대한 알고리즘은 utils.py에서 확인할 수 있다.
			"""
			# Valid the model for every $valid_epoch$ epoch
			if (epoch+1) % args.valid_epochs == 0:
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					gnn_auc_val, gnn_recall_val, gnn_f1_val = test_sage(idx_test, y_test, gnn_model, args.batch_size, self.ckp, flag="val")
					if gnn_auc_val > auc_best:
						gnn_recall_best, f1_mac_best, auc_best, ep_best = gnn_recall_val, gnn_f1_val, gnn_auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				# PC-GNN을 학습할 경우의 test!
				else:
					print("Valid at epoch {}".format(epoch))
					gnn_auc_val, gnn_recall_val, gnn_f1_val = test_pcgnn(idx_test, y_test, gnn_model, args.batch_size, self.ckp, flag="val")
					if gnn_auc_val > auc_best:
						gnn_recall_best, f1_mac_best, auc_best, ep_best = gnn_recall_val, gnn_f1_val, gnn_auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)

		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))
		if args.model == 'SAGE' or args.model == 'GCN':
			gnn_auc, gnn_recall, gnn_f1 = test_sage(idx_test, y_test, gnn_model, args.batch_size, self.ckp, flag="test")
		else:
			gnn_auc, gnn_recall, gnn_f1 = test_pcgnn(idx_test, y_test, gnn_model, args.batch_size, self.ckp, flag="test")
		return gnn_auc, gnn_recall, gnn_f1