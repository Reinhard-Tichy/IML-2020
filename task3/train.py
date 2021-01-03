# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from config.config import BertConfig
import time 
from utils import *

def train(config : BertConfig, model, train_iter, dev_iter, test_iter = None):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_batch = 0 
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 判断是否很久没有效果提升
    model.train()
    CE = nn.CrossEntropyLoss(reduction="none")
    for epoch in range(config.num_epochs):
        train_iter.shuffle()
        trues = []
        predics = []
        logging('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = torch.sum(CE(outputs, labels))
            loss.backward()
            optimizer.step()
            trues.append(labels.data.cpu())
            predics.append(torch.max(outputs.data, 1)[1].cpu())
            if total_batch % config.period == 0:
                # 每多少轮输出在训练集和验证集上的效果
                train_acc = 0.0
                for true,predic in zip(trues, predics):
                    train_acc += metrics.accuracy_score(true, predic)
                train_acc /= len(trues)
                dev_acc, dev_loss, dev_report = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Epoch [{}/{}] Iter: {:>6},  Train Loss: {:>5.3},  Train Acc: {:>6.2%},  Val Loss: {:>5.3},  Val Acc: {:>6.2%},  Time: {} {}s'
                logging(msg.format(epoch + 1, config.num_epochs, total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                logging("Precision, Recall and F1-Score...")
                logging(dev_report)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improve:
                logging("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    if test_iter != None:
        predict(config, model, test_iter)

def predict(config, model, data_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    predict_all = []
    total_batch = 0 
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all += [(int(l),int(p)) for l,p in zip(labels, predic)]
            total_batch += 1
            if total_batch % config.period == 0:
                time_dif = get_time_dif(start_time)
                logging("Iter: {:>6} Time: {}s".format(total_batch, time_dif))
    time_dif = get_time_dif(start_time)
    logging("Time usage: {}s".format(time_dif))
    return predict_all

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, 
                labels=config.id_list, target_names=config.class_list, digits=4)
    return acc, loss_total / len(data_iter), report