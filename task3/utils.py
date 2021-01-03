# coding: UTF-8
import time
import os
import csv
from datetime import timedelta

data_path = "./data"
model_name = "Bert"

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
            f_log.write(s)
            f_log.write('\n')

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def gen_submit(preFile="submission.csv", sampleFile="submission_o.csv"):
    ansDict = {}
    with open(os.path.join(data_path, preFile), "r", encoding='utf-8') as rf:
        sr = csv.reader(rf)
        for row in sr:
            ansDict[row[0]] = row[1]
    title = None
    ansList = []
    with open(os.path.join(data_path, sampleFile), "r", encoding='utf-8') as rf:
        sr = csv.reader(rf)
        for row in sr:
            if row[0] == "" :
                title = row 
                continue
            ansList.append((row[0], ansDict[row[0]]))
    with open(os.path.join(data_path, preFile), "w", encoding='utf-8', newline='') as wf:
        sw = csv.writer(wf)
        sw.writerow(title)
        sw.writerows(ansList)

