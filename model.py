import argparse
import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from load import load_cla_data
from evaluator import evaluate


# helpers: tf.losses.hinge_loss -> F.hinge_embedding_loss
#          tf.losses.log_loss -> F.binary_cross_entropy

class Attention(nn.Module):
    def __init__(self, num_units):
        super(Attention, self).__init__()
        self.av_W = nn.Linear(in_features=num_units, out_features=num_units, bias=True)
        # self.av_u = nn.Linear(in_features=num_units, out_features=1, bias=False)
        self.av_u = nn.Parameter(data=torch.zeros(4), requires_grad=True)
        nn.init.uniform_(self.av_u)
        self.att_weights = nn.Parameter(data=torch.FloatTensor(), requires_grad=True)

        for weight in self.att_weights:
            nn.init.xavier_uniform(tensor=weight)

    def forward(self, hidden_states):
        self.a_linear = self.av_W(hidden_states)
        self.a_laten = torch.tanh(self.a_linear)
        # self.a_scores = self.av_u(self.a_laten)
        self.a_scores = torch.tensordot(self.a_laten, self.av_u, dims=1)
        self.a_alphas = F.softmax(self.a_scores, dim=-1)
        self.a_con = torch.sum(hidden_states * torch.unsqueeze(self.a_alphas, -1), dim=1)
        self.fea_con = torch.cat((hidden_states[:, -1, :], self.a_con), dim=1)
        return self.fea_con


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        if not m.bias == None:
            nn.init.zeros_(m.bias.data)


class LSTM(nn.Module):
    def __init__(self, data_path, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=156, gpu=False, tra_date='2014-01-02', val_date='2015-08-03', tes_date='2015-10-01',
                 att=0, hinge=0, fix_init=0, adv=0, reload=0):
        super(LSTM, self).__init__()
        self.data_path = data_path
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
        self.tra_pv, self.tra_wd, self.tra_gt, \
        self.val_pv, self.val_wd, self.val_gt, \
        self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
            self.data_path, tra_date, val_date, tes_date, seq=self.paras['seq']
        )
        self.fea_dim = self.tra_pv.shape[2]
        self.in_lat = nn.Linear(in_features=self.fea_dim, out_features=self.fea_dim)  # out_features=self.paras['seq']
        # self.lstm_cell = nn.LSTMCell(input_size=self.fea_dim,hidden_size=self.paras['unit'])
        self.outputs_lstm = nn.LSTM(input_size=self.fea_dim,
                                    hidden_size=self.paras['unit'])  # input_size=self.paras['seq']
        if self.att:
            self.attn_layer = Attention(num_units=self.paras['unit'])
            self.attn_layer.apply(initialize_weights)
        self.linear_no_adv = nn.Linear(in_features=self.paras['unit'] * 2, out_features=1)
        self.adv_layer = Adversarial(eps=self.paras['eps'], hinge=self.hinge, att=self.att, unit=self.paras['unit'])
        self.adv_layer.apply(initialize_weights)
        self.pred = None
        self.adv_pred = None

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], self.tra_wd[sta_ind:end_ind, :, :], self.tra_gt[sta_ind:end_ind, :]

    def forward(self, pv_var, wd_var, gt_var):
        pv_var, wd_var, gt_var = torch.from_numpy(pv_var).float(), torch.from_numpy(wd_var).float(), torch.from_numpy(
            gt_var).float()
        feature_mapping = self.in_lat(pv_var)
        outputs, final_states = self.outputs_lstm(feature_mapping)
        if self.att:
            self.fea_con = self.attn_layer(outputs)
            if self.adv_train:
                self.pred, self.adv_pred = self.adv_layer(self.fea_con, gt_var)
            else:
                self.pred = self.linear_no_adv(self.fea_con)
        else:
            if self.adv_train:
                self.pred, self.adv_pred = self.adv_layer(outputs[:, -1, :], gt_var)
        if self.hinge:
            self.loss = F.hinge_embedding_loss(input=self.pred, target=gt_var)
        else:
            self.pred = F.sigmoid(self.pred)
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
            self.fc_W = nn.Linear(in_features=unit, out_features=1, bias=False) # previously named linear_layer

    def forward(self, adv_input, gt_var):
        pred = self.get_pred(adv_input)
        if self.hinge:
            pred_loss = F.hinge_embedding_loss(input=pred, target=gt_var)
        else:
            pred_loss = F.binary_cross_entropy(input=pred, target=gt_var)
        adv_input.retain_grad()
        pred_loss.backward(retain_graph=True)
        grad = adv_input.grad.detach()
        grad = F.normalize(grad, dim=1) #maybe 0?
        self.adv_pv_var = adv_input + self.eps * grad
        self.adv_pred = self.get_pred(self.adv_pv_var)
        if self.hinge:
            self.adv_loss = F.hinge_embedding_loss(input=self.adv_pred, target=gt_var)
        else:
            self.adv_loss = F.binary_cross_entropy(input=self.adv_pred, target=gt_var)
        # adv_pred = e_adv <-> AE
        return pred, self.adv_pred

    def get_pred(self, adv_input):
        if self.att:
            if self.hinge:
                pred = self.fc_W(adv_input)
            else:
                pred = F.sigmoid(self.fc_W(adv_input))
        else:
            if self.hinge:
                pred = self.linear_layer(adv_input)
            else:
                pred = F.sigmoid(self.linear_layer(adv_input))
        return pred
