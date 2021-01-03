from operator import ne
import pandas as pd 
import os 

from src.encoder import Encoder
from src.models import *

data_path = "./data"
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test_sample.csv")
out_path = os.path.join(data_path, "submission.csv")
test_df = pd.read_csv(test_path, sep=',')
train_df = pd.read_csv(train_path, sep=',')

modelDict = {
    "LR" : LR_model,
    "PolyLR" : PolyLR_model,
    "Forest" : forest_model,
    "SVR" : SVR_model, # extremely low
    "MLP" : MLP_model,
    "Bagging" : Bagging_model, # large para size with relatively lower acc
    "GBM" : GBM_model,
    "XGboost" : XGboost_model # not supported on windows
}

def save(data : pd.DataFrame, y, encoder : Encoder, save_path = out_path):
    data['charges'] = y
    if encoder: encoder.rev_transform(data)
    data.to_csv(save_path, sep=',', index=False)

dropAttr = [
    'sex_female',
    'sex_male',
    'smoker_no',
    'smoker_yes',
    'region_northeast',
    'region_northwest',
    'region_southeast',
    'region_southwest'

]

def run_ensemble(train_df):
    encoder = None
    encoder = Encoder(train_df)
    encoder.transform(train_df)
    estimators = []
    scores = []
    labels = []
    nums = list(range(1,5,1)) + list(range(5,60,5)) + list(range(60,100,10))  + list(range(100,500,50))
    for n in nums:
        lr = modelDict["GBM"](n_estimators=n)
        n_train_df = pd.get_dummies(train_df)
        train_score, val_score = lr.train(n_train_df)
        scores += [train_score, val_score]
        estimators += [n,n]
        labels += ['train', 'val']
    return scores, labels, estimators

def run(train_df, test_df):
    encoder = None
    encoder = Encoder(train_df)
    lr = modelDict["GBM"](need_scale=False)
    encoder.transform(train_df)
    n_train_df = pd.get_dummies(train_df)
    lr.train(n_train_df)
    encoder.transform(test_df)
    n_test_df = pd.get_dummies(test_df)
    y = lr.test(n_test_df)
    save(test_df, y, encoder)

if __name__ == "__main__":
    run(train_df, test_df)
