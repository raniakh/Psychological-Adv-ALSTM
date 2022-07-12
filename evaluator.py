import math
import numpy as np
import scipy.stats as sps
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error


def evaluate(prediction, ground_truth, hinge=False, reg=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    if reg:
        performance['mse'] = mean_squared_error(np.squeeze(ground_truth), np.squeeze(prediction))
        return performance

    if hinge:
        pred = (torch.sign(prediction) + 1) / 2
        for ind, p in enumerate(pred):
            v = p.item()
            if abs(v - 0.5) < 1e-8 or np.isnan(v):
                pred[ind][0] = 0
    else:
        pred = torch.round(prediction)
    try:
        # pred[pred == 0] = -1    ## Enable for hinge loss, Disable for BCE
        performance['acc'] = accuracy_score(ground_truth, pred.data.numpy())
    except Exception:
        np.savetxt('prediction', pred.data.numpy(), delimiter=',')
        exit(0)
    performance['mcc'] = matthews_corrcoef(ground_truth, pred.detach().numpy())
    return performance


def compare(current_performance, origin_performance):
    is_better = {}
    for metric_name in origin_performance.keys():
        if metric_name == 'mse':
            if current_performance[metric_name] < \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
        else:
            if current_performance[metric_name] > \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
    return is_better