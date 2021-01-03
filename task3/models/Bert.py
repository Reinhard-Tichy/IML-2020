# coding: UTF-8
from config.config import BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F 
from transformers import BertModel, BertTokenizer

class Model(nn.Module):

    def __init__(self, config : BertConfig, model_name):
        super(Model, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, inplace=False)
        self.fc = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, x):
        '''
            input: x -> [token_ids, seq_len, mask] ids_size:(b,len,hidden_size)
        '''
        context = x[0]
        mask = x[2]
        hidden_states = self.bert(context, attention_mask=mask)[0] # (b, max_len, hidden_size)
        max_hs = self.maxPoolingWithMask(hidden_states, mask)
        hs = self.dropout(max_hs)
        out = self.fc(hs)
        return out

    def maxPoolingWithMask(self, content : torch.LongTensor, mask : torch.LongTensor):
        # return: (b, hidden_size)
        mask = (1-mask) * 1e6
        mask = mask.unsqueeze(-1).expand_as(content)
        return torch.max(content-mask, axis=1)[0]


    def avgPoolingWithMask(self, content : torch.LongTensor, mask : torch.LongTensor):
        # return: (b, hidden_size)
        mask = mask.unsqueeze(-1).expand_as(content)
        return torch.sum(content * mask, axis = 1) / torch.sum(mask, axis=1)

