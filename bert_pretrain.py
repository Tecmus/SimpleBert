from modeling import Bert

import torch
import torch.nn as nn
from collections import OrderedDict
from utils.config_utils import load_config

class BertPretrain(nn.Module):
    def __init__(self, config_path):
        super(BertPretrain, self).__init__()
        self.config = load_config(config_path)

        self.bert = Bert(config_path)

        self.transform = self.transform_layer(self.config.hidden_size)
        self.out_bias = nn.Parameter(torch.zeros(self.config.vocab_size), requires_grad=True)
        self.out_weights = nn.Parameter(self.bert.get_vocab_embedding(), requires_grad=False)

        self.predictions = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.predictions.weight = self.out_weights
        self.predictions.bias = self.out_bias

        self.seq_relationship = nn.Linear(self.config.hidden_size, 2)

    def transform_layer(self, hidden_size):

        return nn.Sequential(
            OrderedDict([
                ("dense", nn.Linear(hidden_size, hidden_size)),
                ("layer_norm", nn.LayerNorm(hidden_size))
            ])
        )
    
    def forward(self, input_ids, position_ids, segment_ids,attention_mask):
        seq_out = self.bert(input_ids, position_ids, segment_ids, attention_mask)
        vocab_embedding = self.bert.get_vocab_embedding()
        vocab_embedding = torch.transpose(vocab_embedding, -1, -2)
        transform_out = self.transform(input_embedding, vocab_embedding)
        logits = self.predictions(transform_out)
        word_probs=nn.Softmax(logits)
        
        return word_probs

def main():
    config_path=''
    model=BertPretrain(config_path)
    seq_logits=model(input)
    
if __name__ == '__main__':
    main()
    