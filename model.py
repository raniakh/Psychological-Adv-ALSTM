import argparse
import copy
import numpy as np
import os
import random
import sys
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from load import load_cla_data, load_cla_data_shuffle
from evaluator import evaluate
# from lossImpl import hingeloss
from sklearn.metrics import hinge_loss


# helpers: tf.losses.hinge_loss -> F.hinge_embedding_loss
#          tf.losses.log_loss -> F.binary_cross_entropy

class Attention(nn.Module):
    def __init__(self, num_units):
        super(Attention, self).__init__()
        self.av_W = nn.Linear(in_features=num_units, out_features=num_units, bias=True)
        # self.av_u = nn.Linear(in_features=num_units, out_features=1, bias=False)
        self.av_u = nn.Parameter(data=torch.empty(num_units), requires_grad=True)  # before data=torch.zeros(4)
        # self.att_weights = nn.Parameter(data=torch.FloatTensor(), requires_grad=True)
        nn.init.uniform_(self.av_u)
        # for weight in self.att_weights:
        #     nn.init.xavier_uniform(tensor=weight)
        self.av_W.apply(initialize_weights)

    def forward(self, hidden_states):
        a_linear = self.av_W(hidden_states)
        a_laten = torch.tanh(a_linear)
        a_scores = torch.tensordot(a_laten, self.av_u, dims=1)  # shape=(None,5)?
        a_alphas = F.softmax(a_scores, dim=-1)
        a_con = torch.sum(hidden_states * torch.unsqueeze(a_alphas, -1), dim=1)
        self.fea_con = torch.cat((hidden_states[:, -1, :], a_con),
                                 dim=1)  # hidden_states[:, -1, :] => last hidden state
        return self.fea_con


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        if not m.bias == None:
            nn.init.zeros_(m.bias.data)


class LSTM(nn.Module):
    def __init__(self, data_path, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=156, gpu=False, tra_date='2014-01-02', val_date='2015-08-03', tes_date='2015-10-01',
                 att=0, hinge=0, fix_init=0, adv=0, reload=0, shuffle=0):
        super(LSTM, self).__init__()
        self.data_path = data_path
        print('self.data_path= ', self.data_path)
        self.model_path = model_path
        self.model_save_path = model_save_path
        # model parameters
        self.paras = copy.copy(parameters)
        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        if att == 1:
            self.att = True
        else:
            self.att = False
        if hinge == 1:
            self.hinge = True
        else:
            self.hinge = False
        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False
        if adv == 1:
            self.adv_train = True
        else:
            self.adv_train = False
        if reload == 1:
            self.reload = True
        else:
            self.reload = False

        # load data
        self.tra_date = tra_date
        self.val_date = val_date
        self.tes_date = tes_date
        if shuffle == 0:
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
                self.data_path, tra_date, val_date, tes_date, seq=self.paras['seq'], hinge=hinge
            )
        else:
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data_shuffle(
                self.data_path, tra_date, val_date, tes_date, seq=self.paras['seq'], hinge=hinge
            )
        self.fea_dim = self.tra_pv.shape[2]
        print('self.fea_dim= ', self.fea_dim)
        print('self.tra_pv,shape= ', self.tra_pv.shape)
        if self.data_path == './data/stocknet-dataset/price/preprocessed_rania_embeddings' \
                or self.data_path == './data/kdd17/preprocessed_rania_embeddings' \
                or self.data_path == '/workspace/ALSTM/data/stocknet-dataset/price/preprocessed_rania_embeddings' \
                or self.data_path == '/workspace/ALSTM/data/kdd17/preprocessed_rania_embeddings':
            self.in_pr = nn.Linear(in_features=6, out_features=6)
            self.in_gm = nn.Linear(in_features=6, out_features=6)
            self.in_rest = nn.Linear(in_features=self.fea_dim-12, out_features=self.fea_dim-12)

        self.in_lat = nn.Linear(in_features=self.fea_dim, out_features=self.fea_dim)  # out_features=self.paras['seq']
        # self.lstm_cell = nn.LSTMCell(input_size=self.fea_dim,hidden_size=self.paras['unit'])
        ## TRY #OVERFIT TRAINING
        # self.in_lat_2 = nn.Linear(in_features=self.fea_dim+10, out_features=self.fea_dim+2)
        ##
        self.outputs_lstm = nn.LSTM(input_size=self.fea_dim,  # (+2) TRY #OVERFIT TRAINING
                                    hidden_size=self.paras['unit'], batch_first=True)  # input_size=self.paras['seq']
        if self.att:
            self.attn_layer = Attention(num_units=self.paras['unit'])
        self.linear_no_adv = nn.Linear(in_features=self.paras['unit'] * 2, out_features=1)
        self.adv_layer = Adversarial(eps=self.paras['eps'], hinge=self.hinge, att=self.att, unit=self.paras['unit'])
        self.pred = None
        self.adv_pred = None
        self.in_lat.apply(initialize_weights)
        self.linear_no_adv.apply(initialize_weights)

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], self.tra_wd[sta_ind:end_ind, :, :], self.tra_gt[sta_ind:end_ind, :]

    def forward(self, pv_var, wd_var, gt_var, f):
        pv_var, wd_var, gt_var = torch.from_numpy(pv_var).float(), torch.from_numpy(wd_var).float(), torch.from_numpy(
            gt_var).float()

        if self.data_path == './data/stocknet-dataset/price/preprocessed_rania_embeddings' \
                or self.data_path == './data/kdd17/preprocessed_rania_embeddings' \
                or self.data_path == '/workspace/ALSTM/data/stocknet-dataset/price/preprocessed_rania_embeddings' \
                or self.data_path == '/workspace/ALSTM/data/kdd17/preprocessed_rania_embeddings':
            # preprocess_rania - concatenation + in_lat
            original_features_mapping = self.in_rest(pv_var[:, :, 0:11])
            gm_feature_mapping = self.in_gm(pv_var[:, :, 11:17])
            pr_feature_mapping = self.in_pr(pv_var[:, :, 17:])
            feature_concat = torch.cat((original_features_mapping, gm_feature_mapping, pr_feature_mapping), dim=2)
            feature_mapping_tmp = self.in_lat(feature_concat)
        else:
            # original code
            feature_mapping_tmp = self.in_lat(pv_var)
        feature_mapping = torch.tanh(feature_mapping_tmp)  # Added 08.03.22
        outputs, final_states = self.outputs_lstm(feature_mapping)
        if self.att:
            self.fea_con = self.attn_layer(outputs)
            if self.adv_train:
                self.pred, self.adv_pred = self.adv_layer(self.fea_con, gt_var) # TODO: - the two outputs are the same up to a constant, is this ok?
            else:
                self.pred = self.linear_no_adv(self.fea_con)
        else:
            if self.adv_train:
                self.pred, self.adv_pred = self.adv_layer(outputs[:, -1, :], gt_var)
        if self.hinge:
            lossfn = nn.MarginRankingLoss(margin=1)
            zeros = torch.zeros_like(self.pred)
            self.loss = lossfn(input1=self.pred, input2=zeros, target=gt_var)
        else:
            self.pred = torch.sigmoid(self.pred)
            self.loss = F.binary_cross_entropy(input=self.pred, target=gt_var)
        return self.pred, self.adv_pred


class Adversarial(nn.Module):
    def __init__(self, eps, hinge, att, unit):
        self.eps = eps
        self.att = att
        super(Adversarial, self).__init__()
        self.hinge = hinge
        if self.att:
            self.fc_W = nn.Linear(in_features=unit * 2, out_features=1, bias=True)
        else:
            self.fc_W = nn.Linear(in_features=unit, out_features=1, bias=False)  # previously named linear_layer
        self.fc_W.apply(initialize_weights)
        self.adv_loss = 0.0

    def forward(self, adv_input, gt_var):
        pred = self.get_pred(adv_input)
        if self.hinge:
            lossfn = nn.MarginRankingLoss(margin=1)
            zeros = torch.zeros_like(pred)
            pred_loss = lossfn(input1=pred, input2=zeros, target=gt_var)
        else:
            pred_loss = F.binary_cross_entropy(input=pred, target=gt_var)
        adv_input.retain_grad()
        pred_loss.backward(retain_graph=True)
        grad = adv_input.grad.detach()
        grad = F.normalize(grad, dim=1)  # maybe 0?
        self.adv_pv_var = adv_input + self.eps * grad
        self.adv_pred = self.get_pred(self.adv_pv_var)
        if self.hinge:
            lossfn = nn.MarginRankingLoss(margin=1)
            zeros = torch.zeros_like(self.adv_pred)
            self.adv_loss = lossfn(input1=self.adv_pred, input2=zeros, target=gt_var)
        else:
            self.adv_loss = F.binary_cross_entropy(input=self.adv_pred, target=gt_var)
        # adv_pred = e_adv <-> AE
        return pred, self.adv_pred

    def get_pred(self, adv_input):
        if self.att:
            if self.hinge:
                pred = self.fc_W(adv_input)
            else:
                pred = torch.sigmoid(self.fc_W(adv_input))
        else:
            if self.hinge:
                pred = self.linear_layer(adv_input)
            else:
                pred = torch.sigmoid(self.linear_layer(adv_input))
        return pred
