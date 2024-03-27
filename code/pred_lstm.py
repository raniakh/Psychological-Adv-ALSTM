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


class AWLSTM(nn.Module):
    def __init__(self, data_path, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=156, gpu=False, tra_date='2014-01-02', val_date='2015-08-03', tes_date='2015-10-01',
                 att=0, hinge=0, fix_init=0, adv=0, reload=0):
        super().__init__()
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

        # layers
        self.in_lat = nn.Linear(in_features=self.paras['seq'], out_features=self.fea_dim)
        self.outputs = nn.LSTM(input_size=self.fea_dim, hidden_size=self.paras['unit'])
        # attention
        if self.att:
            self.av_W = nn.Linear(in_features=self.paras['unit'], out_features=self.paras['unit'], bias=True)
            self.av_u = nn.Linear(in_features=self.paras['unit'], out_features=1, bias=False)
            # adversarial
            self.fc_W = nn.Linear(in_features=self.paras['unit']*2, out_features=1, bias=True)
        else:
            self.pred_adv = nn.Linear(in_features=self.paras['unit']*2, out_features=1)

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], self.tra_wd[sta_ind:end_ind, :, :], self.tra_gt[sta_ind:end_ind, :]

    # pv_var shape(batch_size, history window length, 11 features)
    # wd_var shape(batch_size, history window length, 5)
    # gt_var shaoe(batch_size, 1) - true labels
    def forward(self, pv_var, wd_var, gt_var):
        if self.att:
            feature_mapping = self.in_lat(pv_var)
            hidden_states = self.outputs(feature_mapping)
            hidden_states = torch.FloatTensor(hidden_states) #TODO check
            a_linear = self.av_W(hidden_states)
            self.a_laten = F.tanh(a_linear)
            self.a_scores = self.av_u(self.a_laten)
            self.a_alphas = F.softmax(self.a_scores)
            self.a_con = torch.sum(self.outputs * torch.unsqueeze(self.a_alphas, -1), dim=1)
            self.fea_con = torch.cat((self.outputs[:, -1, :], self.a_con), dim=1)
            if self.hinge:
                pred = self.fc_W(self.fea_con)
            else:
                pred = F.sigmoid(self.fc_W(self.fea_con))
        else:
            if self.hinge:
                pred = self.fc_W(self.outputs[:, -1, :])
            else:
                pred = F.sigmoid(self.fc_W(self.outputs[:, -1, :]))

    def train(self, tune_para=False):
        if self.reload:
            self.load_state_dict(torch.load(self.model_path))
            print('model restored')

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        best_test_perf = {
            'acc': 0, 'mcc': -2
        }

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        for i in range(self.epochs):
            t1 = time()
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_adv = 0.0
            for j in range(bat_count):
                pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.wd_var: wd_b,
                    self.gt_var: gt_b
                }




if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./data/stocknet-dataset/price/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int, default=5)
    parser.add_argument('-u', '--unit', help='number pf hidden units in lstm', type=int, default=32)
    parser.add_argument('-l2', '--alpha_l2', type=float, help='alpha for l2 regularizer', default=1e-2)
    parser.add_argument('-la', '--beta_adv', type=float, help='beta for adversarial loss', default=1e-2)
    parser.add_argument('-le', '--epsilon_adv', type=float, help='epsilon to control the scale of noise',
                        default=1e-2)
    parser.add_argument('-s', '--step', help='steps to make prediction', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate', type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-q', '--model_path', help='path to load model', type=str,
                        default='./saved_model/acl18_alstm/exp')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model', default='./tmp/model')
    parser.add_argument('-o', '--action', type=str, help='train,test,pred', default='train')
    parser.add_argument('-m', '--model', type=str, help='pure_lstm, di_lstm, att_lstm, week_lstm, aw_lstm',
                        default='pure_lstm')
    parser.add_argument('-f', '--fix_init', type=int, help='use fixed initializaton', default=0)
    parser.add_argument('-a', '--att', type=int, help='use attention model', default=1)
    parser.add_argument('-w', '--week', type=int, help='use week day data', default=0)
    parser.add_argument('-v', '--adv', type=int, help='adversarial training', default=0)
    parser.add_argument('-hi', '--hinge_loss', type=int, help='use hinge loss', default=1)
    parser.add_argument('-rl', '--reload', type=int, help='use pre-trained parameters', default=0)
    args = parser.parse_args()
    print(args)

    parameters = {
        'seq': int(args.seq),
        'unit': int(args.unit),
        'alp': float(args.alpha_l2),
        'bet': float(args.beta_adv),
        'eps': float(args.epsilon_adv),
        'lr': float(args.learning_rate)
    }

    if 'stocknet' in args.path:
        tra_date = '2014-01-02'
        val_date = '2015-08-03'
        tes_date = '2015-10-01'
    elif 'kdd17' in args.path:
        tra_date = '2007-01-03'
        val_date = '2015-01-02'
        tes_date = '2016-01-04'
    else:
        print('unexpected path: %s' % args.path)
        exit(0)

    pure_LSTM = AWLSTM(
        data_path=args.path,
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
        tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
        hinge=args.hinge_loss, fix_init=args.fix_init, adv=args.adv, reload=args.reload
    )

    if args.action == 'train':
        pure_LSTM.train()
    elif args.action == 'test':
        print('pure_LSTM.test')
    elif args.action == 'report':
        for i in range(5):
            pure_LSTM.train()
    elif args.action == 'pred':
        print('pure_LSTM.predict_record()')
    elif args.action == 'adv':
        print('pure_LSTM.predict_adv()')
    elif args.action == 'latent':
        print('pure_LSTM.get_latent_rep()')
