import argparse
import copy
import numpy as np
import os
import pickle
import random
from sklearn.utils import shuffle
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from load import load_cla_data
from evaluator import evaluate
from model import LSTM, initialize_weights


def train(model, optimizer, tune_para=False):
    model.train()
    if model.reload:
        model.load_state_dict(torch.load(model.model_path))

    best_valid_pred = np.zeros(model.val_gt.shape, dtype=float)
    best_test_pred = np.zeros(model.tes_gt.shape, dtype=float)

    best_valid_perf = {
        'acc': 0, 'mcc': -2
    }
    best_test_perf = {
        'acc': 0, 'mcc': -2
    }
# TODO add loss + obj function in train
    bat_count = model.tra_pv.shape[0] // model.batch_size
    if not (model.tra_pv.shape[0] % model.batch_size == 0):
        bat_count += 1
    for i in range(model.epochs):
        print('-- EPOCH {}:'.format(i), file=f)
        t1 = time()
        tra_loss = 0.0
        tra_obj = 0.0
        l2 = 0.0
        tra_adv = 0.0
        for j in range(bat_count):
            optimizer.zero_grad()
            print('-- BATCH {}:'.format(j), file=f)
            print('-- TRAINING', file=f)
            pv_b, wd_b, gt_b = model.get_batch(j * model.batch_size)
            pred, adv_pred = model(pv_b, wd_b, gt_b, f)
            # pred, adv_pred = torch.sign(pred), torch.sign(adv_pred) # TODO do i need to take sign?
            tra_vars = [model.adv_layer.fc_W.weight, model.adv_layer.fc_W.bias]
            for var in tra_vars:
                l2 += torch.sum(var**2)/2
            loss = model.loss + parameters['bet']*model.adv_layer.adv_loss + parameters['alp']*l2
            loss.backward(retain_graph=True)
            optimizer.step()
            tra_loss += model.loss
            tra_obj += loss.data
            tra_adv += model.adv_layer.adv_loss
        print('----->>>>> Training:', (tra_obj / bat_count).item(), (tra_loss / bat_count).item(),
              (l2 / bat_count).item(), (tra_adv / bat_count).item())
        if not tune_para:
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_acc = 0.0
            for j in range(bat_count):
                pv_b, wd_b, gt_b = model.get_batch(j * model.batch_size)
                pred, adv_pred = model(pv_b, wd_b, gt_b, f)
                cur_tra_perf = evaluate(adv_pred, gt_b, model.hinge)
                tra_loss += model.loss
                tra_vars = [model.adv_layer.fc_W.weight, model.adv_layer.fc_W.bias]
                for var in tra_vars:
                    l2 += torch.sum(var ** 2) / 2
                loss = model.loss + parameters['bet'] * model.adv_layer.adv_loss + parameters['alp'] * l2
                tra_obj += loss.data
                loss.backward(retain_graph=True)
                optimizer.step()
                tra_acc += cur_tra_perf['acc']
            print('Training:', (tra_obj / bat_count).item(), (tra_loss / bat_count).item(),
                  (l2 / bat_count).item(), (tra_acc / bat_count).item())
        # test on validation
        print('---->>>>> Validation')
        print('-- VALIDATION', file=f)
        val_pred, val_adv_pred = model(model.val_pv, model.val_wd, model.val_gt, f)
        cur_valid_perf = evaluate(val_pred, model.val_gt, model.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', model.loss.item())
        print('---->>>>> Testing')
        print('-- TESTING', file=f)
        test_pred, test_adv_pred = model(model.tes_pv, model.tes_wd, model.tes_gt, f)
        cur_test_perf = evaluate(test_pred, model.tes_gt, model.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', model.loss.item())

        if cur_valid_perf['acc'] > best_valid_perf['acc']:
            best_valid_perf = copy.copy(cur_valid_perf)
            best_valid_pred = copy.copy(val_pred)
            best_test_perf = copy.copy(cur_test_perf)
            best_test_pred = copy.copy(test_pred)
            if not tune_para:
                torch.save(model.state_dict(), model.model_save_path)
        model.tra_pv, model.tra_wd, model.tra_gt = shuffle(
            model.tra_pv, model.tra_wd, model.tra_gt, random_state=0
        )

        t4 = time()
        print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
    print('\nBest Valid performance:', best_valid_perf)
    print('\tBest Test performance:', best_test_perf)


if __name__ == '__main__':
    f = open('checkInputsAtDifferentStages_debugging.txt', 'w')
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

    lstm = LSTM(
        data_path=args.path,
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
        tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
        hinge=args.hinge_loss, fix_init=args.fix_init, adv=args.adv, reload=args.reload
    )

    pytorch_total_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
    print(pytorch_total_params)

    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    lstm.to(device)
    lstm.apply(initialize_weights)
    optimizer = optim.Adam(lstm.parameters(), lr=parameters['lr'])
    if args.action == 'train':
        train(model=lstm, optimizer=optimizer)

    f.close()