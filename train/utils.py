import numpy as np
import argparse
import os
import pathlib
import yaml
import torch
import random

def load_config(path=None):
    if path is None:
        path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        'train_configs/train_config.yaml')
        
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_configs/train_config.yaml')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    return args

def set_all_seeds(seed):
    """Set all the seeds to fix the same initial conditions for each training;
    :param seed: seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def mean_squared_error(y_true, y_pred):
    """
    计算MSE
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def r_squared(y_true, y_pred):
    """
    计算R平方
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def explained_variance_score(y_true, y_pred):
    """
    计算解释方差分数
    """
    var_res = np.var(y_true - y_pred)
    var_tot = np.var(y_true)
    explained_variance = 1 - (var_res / var_tot)
    return explained_variance

def custom_accuracy(y_true, y_pred, threshold=0.01):
    """
    计算自定义准确率。
    y_pred: 预测值，tensor
    y_true: 真实值，tensor
    threshold: 阈值百分比，表示允许的误差范围，默认为1%
    """
    # 计算真实值的上下界
    lower_bound = y_true * (1 - threshold)
    upper_bound = y_true * (1 + threshold)
    
    # 检查预测值是否在允许的误差范围内
    correct_predictions = (y_pred >= lower_bound) & (y_pred <= upper_bound)
    
    # 计算在误差范围内的预测值的比例
    accuracy = np.mean(correct_predictions.astype(float))
    
    return accuracy

def percentage_accurate(y_true, y_pred, threshold=0.01):

    lower_bound = y_true * (1 - threshold)
    upper_bound = y_true * (1 + threshold)
    
    return (y_pred >= lower_bound) & (y_pred <= upper_bound)

def mape(actuals, forecasts):
    return np.mean(np.abs((actuals - forecasts) / actuals)) * 100