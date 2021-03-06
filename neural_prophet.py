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
from neuralprophet import NeuralProphet
import pandas as pd 


DATASET_PATH = '/home/v-tyan/Time_Series_Survey/datasets/'
TRAIN_TEST_SPLIT_RANDOM_STATE = 42
SEED = 2021

# seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


X = np.loadtxt(os.path.join(DATASET_PATH, 'M4/M4_sample_1_X.txt'))
y = np.loadtxt(os.path.join(DATASET_PATH, 'M4/M4_sample_1_y.txt'))
print(X.shape, y.shape)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
val_X, test_X, val_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

# for NN methods
train_X, train_y = np.log1p(train_X), np.log1p(train_y)  
val_X, val_y = np.log1p(val_X), np.log1p(val_y)
test_X, test_y = np.log1p(test_X), np.log1p(test_y)

fp = open('result.txt', 'w')

for idx in range(test_X.shape[0]):
    df = pd.DataFrame({'ds': pd.date_range('2007-1-1', '2007-2-19'), 'y': test_X[idx, :]})
    print(df)

    model = NeuralProphet()
    model.fit(df, freq='D')
    future = model.make_future_dataframe(df, periods=1)
    print(future)
    break