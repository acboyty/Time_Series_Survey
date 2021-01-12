import xgboost as xgb 
import lightgbm as lgb 
import catboost as cat 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE 
from rnn import RNNModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from nbeats_pytorch.model import NBeatsNet
import random
from utils import sMAPE, MAPE, run_gbdt, run_nbeats, run_rnn


DATASET_PATH = '/home/v-tyan/Time_Series_Survey/datasets/'
TRAIN_TEST_SPLIT_RANDOM_STATE = 42
SEED = 2021

# seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


X = np.loadtxt(os.path.join(DATASET_PATH, 'M4/M4_ave_X.txt'))
y = np.loadtxt(os.path.join(DATASET_PATH, 'M4/M4_ave_y.txt'))
print(X.shape, y.shape)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

fp = open('result.txt', 'w')

# # xgb

# model = xgb.XGBRegressor(n_estimators=10000)
# test_y_hat = run_gbdt(model, train_X, train_y, val_X, val_y, test_X)
# fp.write(f'xgb >> MAPE: {MAPE(test_y, test_y_hat):.5f}, sMAPE: {sMAPE(test_y, test_y_hat):.5f}\n')

# # lgb

# model = lgb.LGBMRegressor(n_estimators=10000)
# test_y_hat = run_gbdt(model, train_X, train_y, val_X, val_y, test_X)
# fp.write(f'lgb >> MAPE: {MAPE(test_y, test_y_hat):.5f}, sMAPE: {sMAPE(test_y, test_y_hat):.5f}\n')

# # cat

# model = cat.CatBoostRegressor(iterations=10000, verbose=10)
# test_y_hat = run_gbdt(model, train_X, train_y, val_X, val_y, test_X)
# fp.write(f'cat >> MAPE: {MAPE(test_y, test_y_hat):.5f}, sMAPE: {sMAPE(test_y, test_y_hat):.5f}\n')

# for NN methods if the value are too large and > 0
train_X, train_y = np.log1p(train_X), np.log1p(train_y)  
val_X, val_y = np.log1p(val_X), np.log1p(val_y)
test_X, test_y = np.log1p(test_X), np.log1p(test_y)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

train_X, train_y = torch.Tensor(train_X), torch.Tensor(train_y)
train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=1024, shuffle=True)
val_X, val_y = torch.Tensor(val_X), torch.Tensor(val_y)
# val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=128, shuffle=True)
test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y)
# test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=128, shuffle=True)

# nbeats

model = NBeatsNet(device=device, stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), 
    forecast_length=1, backcast_length=50, hidden_layer_units=128, share_weights_in_stack=False)
test_y_hat = run_nbeats(model, train_loader, val_X, val_y, test_X, test_y, fp, device)

# rnn

model = RNNModel().to(device)
test_y_hat = run_rnn(model, train_loader, val_X, val_y, test_X, test_y, fp, device)

fp.close()

# save prediction
np.savetxt('test_X.txt', np.expm1(test_X))
np.savetxt('test_y.txt', test_y)
np.savetxt('test_y_hat.txt', test_y_hat)