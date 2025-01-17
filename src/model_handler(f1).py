import time, datetime
import os
import torch
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test, load_data, pos_neg_split, normalize, pick_step, set_seeds
from src.model import PCALayer
from src.layers import InterAgg1, InterAgg3, InterAgg5, IntraAgg
from src.graphsage import *
from src.result_manager import *

"""
        Training PC-GNN
        Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class ModelHandler(object):

        def __init__(self, config):
                self.result = ResultManager(args=config) # TODO
                args = argparse.Namespace(**config)
                # args.cuda = not args.no_cuda and torch.cuda.is_available()
                # device = torch.device(args.cuda_id)
                # torch.cuda.set_device(device)

                # load graph, feature, and label
                homo, relation_list, feat_data, labels = load_data(args.data_name) # KDK 데이터 셋에서 realation의 수가 달라질 수 있어 수정함.

                # train/validation/test set 분할.
                np.random.seed(args.seed)
                random.seed(args.seed)
                
                if args.data_name.startswith('amazon'):
                        idx_unlabeled = 2013 if args.data_name == 'amazon_new' else 3305
                        # 0-3304 are unlabeled nodes (Unlabeled 노드는 학습과 검증 과정에서 제외함.)
                        index = list(range(idx_unlabeled, len(labels))) # Stratified sampling으로 train/validation/test를 구분한다.
                        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[idx_unlabeled:], stratify=labels[idx_unlabeled:], train_size=args.train_ratio, random_state=args.seed, shuffle=True)
                        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
                
                elif args.data_name == 'yelp':
                        index = list(range(len(labels))) # Stratified sampling을 통해 데이터셋을 train/validation/test set으로 분할한다.
                        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio, random_state=args.seed, shuffle=True)
                        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
                        
                print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
                        f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
                print(f"Feature dimension: {feat_data.shape[1]}")


                # split pos neg sets for under-sampling
                train_pos, train_neg = pos_neg_split(idx_train, y_train)

                # Feature normalization 부분 코드.
                if args.data_name.startswith('amazon'):
                        feat_data = normalize(feat_data)

                # set input graph
                if args.model == 'SAGE' or args.model == 'GCN':
                        adj_lists = homo
                else: # PC-GNN은 multi-relational graph를 이용함.
                        adj_lists = relation_list

                print(f'Model: {args.model}, emb_size: {args.emb_size}.')
                
                self.args = args
                self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
                                                'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
                                                'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
                                                'train_pos': train_pos, 'train_neg': train_neg}


        def train(self):
                args = self.args
                idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
                idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']

                self.model_select()

                model = self.model.cuda()

                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
                auc_best, f1_mac_best, epoch_best = 1e-10, 1e-10, 0

                # train the model
                for epoch in range(args.epochs):
                        if args.model == 'PCGNN':
                                sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos'])*2)
                        else:
                                sampled_idx_train = idx_train
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
                                loss = model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))

                                loss.backward()
                                optimizer.step()
                                end_time = time.time()
                                epoch_time += end_time - start_time
                                loss += loss.item()

                        """
                        Test 과정에 대한 알고리즘은 utils.py에서 확인할 수 있다.
                        """
                        # Valid the model for every $valid_epoch$ epoch
                        if (epoch+1) % args.valid_epochs == 0:
                                print("Valid at epoch {}".format(epoch))
                                auc_val, recall_val, f1_mac_val, precision_val = test(idx_valid, y_valid, model, self.args.batch_size, self.result, epoch, epoch_best, flag="val")
                                gain_auc = (auc_val - auc_best)/auc_best
                                gain_f1_mac =  (f1_mac_val - f1_mac_best)/f1_mac_best
                                if (gain_auc + gain_f1_mac) > 0:
                                        gnn_recall_best, f1_mac_best, auc_best, epoch_best = recall_val, f1_mac_val, auc_val, epoch
                                        torch.save(model.state_dict(), self.result.model_path)
                        if (epoch - epoch_best) > self.args.patience:
                                line = f"Early stopping at epoch {epoch}"
                                print(line)
                                break

                print("Restore model from epoch {}".format(epoch_best))
                model.load_state_dict(torch.load(self.result.model_path))
                auc_test, recall_test, f1_mac_test, precision_test = test(idx_test, y_test, model, self.args.batch_size, self.result, epoch_best=epoch_best, flag="test")
                return auc_test, recall_test, f1_mac_test
        
        def model_select(self):
                args = self.args
                
                # 클래스 변수로부터 feature, label, adj 생성한다.
                feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
                
                # initialize model input
                features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
                features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
                features = features.cuda()
                # build one-layer models
                if args.model == 'SAGE':
                        agg_sage = MeanAggregator(features, cuda=True)
                        enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=True, cuda=True)
                elif args.model == 'GCN':
                        agg_gcn = GCNAggregator(features, cuda=True)
                        enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, cuda=True)

                if args.model == 'PCGNN':
                        if (args.data_name == "yelp") or (args.data_name.startswith('amazon')):
                                intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=True)
                                intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=True)
                                intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=True)
                                inter1 = InterAgg3(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
                                                                adj_lists, [intra1, intra2, intra3], cuda=True)
                        else:
                                intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=True)
                                inter1 = InterAgg1(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
                                                                adj_lists, [intra1], cuda=True)
                        model = PCALayer(2, inter1, args.alpha) # 앞서 정의한 inter/intra aggregator를 이용하여 PCGNN 모델을 생성함.
                elif args.model == 'SAGE':
                        # the vanilla GraphSAGE model as baseline
                        # enc_sage.num_samples = 5
                        model = GraphSage(2, enc_sage)
                elif args.model == 'GCN':
                        model = GCN(2, enc_gcn)
                
                self.model = model