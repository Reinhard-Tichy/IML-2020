# coding: UTF-8
import csv 
import os
from tqdm import tqdm
import torch 
from transformers import BertTokenizer
import random

class BertConfig(object):
    PAD, CLS, SEP = '[PAD]', '[CLS]' ,'[SEP]' # padding符号, bert中综合信息符号

    def __init__(self) -> None:
        
        self.dataPath = "./data" # 数据所在地址
        self.model_name = "bert-base-uncased" # BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.label2id = {
            "positive" : 0,
            "negative" : 1
        }
        self.id2label = {v:k for k,v in self.label2id.items()}
        self.class_list = []
        self.id_list = []
        for k,v in self.label2id.items():
            self.class_list.append(k)
            self.id_list.append(v)
        self.trainFile = "train_data.csv"
        self.devFile = "eval_data.csv"
        self.testFile = "test_data.csv"
        self.outFile = "submission.csv"
        self.save_path = os.path.join('./saved_dict/', self.model_name + '.ckpt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.max_seq_len = 512
        self.require_improve = 3500                                      
        self.class_num = 2                                            
        self.num_epochs = 5                                           
        self.batch_size = 8                                           
        self.learning_rate = 2e-5                                      
        self.hidden_size = 768
        self.period = 100
        self.hidden_dropout_prob = 0.1

        if not os.path.exists("saved_dict"):
            os.mkdir("saved_dict")
        if not os.path.exists("log"):
            os.mkdir("log")

    def load_data(self, dataFileName, test=False):
        '''
            returns: content:list of (token_ids, label_id, seq_len, mask)
        '''
        path = os.path.join(self.dataPath, dataFileName)
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            sr = csv.reader(f)
            for row in tqdm(sr):
                if row[0] == "": continue
                if test:
                    label = row[0] # id
                else : label = self.label2id[row[-1]]
                doc = row[1]
                tokens = self.tokenizer.tokenize(doc)
                seq_len = len(tokens) + 2
                mask = []

                if seq_len < self.max_seq_len:
                    tokens = [self.CLS] + tokens + [self.SEP] + [self.PAD] * (self.max_seq_len - seq_len)
                    mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    mask = [1] * self.max_seq_len
                    tokens = tokens[:self.max_seq_len-2]
                    tokens = [self.CLS] + tokens + [self.SEP]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    seq_len = self.max_seq_len
                contents.append((token_ids, int(label), seq_len, mask))
        return contents
    
    def get_data_iter(self, dataFileName, test=False):
        contents = self.load_data(dataFileName, test)
        return DatasetIterater(contents, self.batch_size, self.device)

    def get_train_eval_test_iter(self):
        train_iter = self.get_data_iter(self.trainFile)
        eval_iter = self.get_data_iter(self.devFile)
        return train_iter, eval_iter

    def get_predict_iter(self):
        test_iter = self.get_data_iter(self.testFile, test=True)
        return test_iter

    def gen_submit(self, predict_all : list):
        with open(self.outFile, "w+", encoding='utf-8') as wf:
            writer = csv.writer(wf)
            for pid, plabelid in predict_all:
                plabel = self.id2label[plabelid]
                writer.writerow((pid, plabel))

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        random.shuffle(self.batches)
        self.n_batches = len(batches) // batch_size
        if len(batches) % self.n_batches != 0:
            self.n_batches += 1
        self.index = 0
        self.device = device

    def shuffle(self):
        random.shuffle(self.batches)

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: min((self.index + 1) * self.batch_size,len(self.batches))]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches
