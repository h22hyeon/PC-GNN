import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step
from src.model import PCALayer
from src.layers import InterAgg, IntraAgg
from src.graphsage import *


"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler(object):

	def __init__(self, config, ckp):
		args = argparse.Namespace(**config)
		args.cuda = not args.no_cuda and torch.cuda.is_available()
		os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

		# load graph, feature, and label
		homo, relation_list, feat_data, labels = load_data(args.data_name, prefix=args.data_dir) # KDK 데이터 셋에서 realation의 수가 달라질 수 있어 수정함.

		# train/validation/test set 분할.
		np.random.seed(args.seed)
		random.seed(args.seed)
		if args.data_name == 'yelp':
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
		if args.model == 'PCGNN' & args.data_name == "KDK":
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra4 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra5 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], adj_lists, [intra1, intra2, intra3, intra4, intra5], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'PCGNN' & args.data_name != "KDK":
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
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

		timestamp = time.time()
		timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
		dir_saver = os.path.join(args.save_dir, timestamp)
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

			# Valid the model for every $valid_epoch$ epoch
			if epoch % args.valid_epochs == 0:
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				else:
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_pcgnn(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)

		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))
		if args.model == 'SAGE' or args.model == 'GCN':
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		else:
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
