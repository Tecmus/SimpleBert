from modeling import Bert

import torch
import torch.nn as nn
from collections import OrderedDict
from bert_pretrain import BertPretrain

class BertClassifier(nn.Module):
    def __init__(self, config_path):
        super(BertClassifier, self).__init__()
        self.config = load_config(config_path)
        pretain_model_path=self.config['pretain_model']
        # self.bert_pretrain = BertPretrain(config_path)
        pretrain_model = BertPretrain(config_path)
        if pretain_model_path:
          
            pretrain_model.load_state_dict(torch.load(pretain_model_path))
            print('pretrain load success')
          
        self.bert=pretrain_model.bert
        
        self.predictions = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self, input_ids, position_ids, segment_ids, attention_mask):
        pooler_out, seq_out = self.bert(input_ids, position_ids, segment_ids, attention_mask)
        logits = self.predictions(pooler_out)
        cls_probs = self.softmax(logits)

        return cls_probs


