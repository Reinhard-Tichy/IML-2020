# coding: UTF-8
import csv 
import os 
from tqdm import tqdm
import random

data_path = "./data"
train_path = os.path.join(data_path, "train_data.csv")
eval_path = os.path.join(data_path, "eval_data.csv")
ori_path = os.path.join(data_path, "train.csv")

def train_eval_split(data_path, train_path, eval_path, k = 0.8):
    title = []
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        sr = csv.reader(f)
        for row in sr:
            if row[0] == "" :
                title = row 
                continue
            data.append([r.strip() for r in row])
    random.shuffle(data)
    train_data = data[ : int(len(data) * k)]
    eval_data = data[int(len(data) * k) : ]
    with open(train_path, "w+", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        writer.writerows(train_data)
    with open(eval_path, "w+", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        writer.writerows(eval_data)

if __name__ == "__main__":
    train_eval_split(ori_path, train_path, eval_path)