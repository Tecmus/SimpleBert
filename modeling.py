#!/home/azureuser/anaconda3/bin/python
# coding:utf-8

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from collections import OrderedDict


def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return nn.GELU()
    else:
        assert False, 'Unknown activation function {}'.format(name)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_head, hidden_dropout_prob, seq_len):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.seq_len = seq_len

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size,eps=1e-12)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, q, k, v, mask):

        assert self.hidden_size % self.num_head == 0
        depth = self.hidden_size // self.num_head

        head_q = self.w_q(q).view(
            (-1, self.seq_len, self.num_head, depth)).permute(0, 2, 1, 3)  # [batch_size,self.num_head,self.seq_len,d_model]

        head_k = self.w_k(k).view(
            (-1, self.seq_len, self.num_head, depth)).permute(0, 2, 1, 3)

        head_v = self.w_v(v).view(
            (-1, self.seq_len, self.num_head, depth)).permute(0, 2, 1, 3)

        attn, _ = self.self_attn(head_q, head_k, head_v, depth, mask)

        attn.permute(0, 2, 1, 3)  # reverse
        attn = attn.view(-1, self.seq_len, self.hidden_size)

        self.num_head_attn = self.dense(attn)
        self.num_head_attn = self.dropout(self.num_head_attn)
        self.num_head_attn = self.layer_norm(q + self.num_head_attn)

        return self.num_head_attn

    def self_attn(self, q, k, v, hidden_size, mask):
        qv_logits = torch.matmul(q, k.permute(
            0, 1, 3, 2)) / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
        
        if  mask is not None:
            qv_logits += (1 - mask) * (-1e9)

        attn_weights = torch.softmax(qv_logits, -1)
        attn = torch.matmul(attn_weights, v)
        return attn, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config

        self.multi_head_attention = MultiHeadAttention(config.hidden_size, config.num_head, config.hidden_dropout_prob, config.seq_len)

        self.feed_forward = self.feed_forward_layer(config.d_model, config.ffn_size)

        self.layer_norm = nn.LayerNorm(config.hidden_size,eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def feed_forward_layer(self, d_model, ffn_size):

        return nn.Sequential(OrderedDict(
            [
                ("intermediate", nn.Linear(d_model, ffn_size)),
                ("out_dense", nn.Linear(ffn_size, d_model)),
                ("activation", get_activation(self.config.hidden_act))
            ]
        )

        )

    def forward(self, input_emb, mask):

        attn_out = self.multi_head_attention(
            input_emb, input_emb, input_emb, mask)
        ffn_out = self.feed_forward(attn_out)

        layer_output = self.layer_norm(ffn_out + attn_out)

        return layer_output


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(self.config.num_hidden_layers)])

    def forward(self, input_emb, encoder_mask):

        for layer in self.layers:
            input_emb = layer(input_emb, encoder_mask)

        return input_emb


class Bert(nn.Module):
    def __init__(self, config_path):
        super(Bert, self).__init__()
        with open(config_path) as f:
            self.config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.segment_embeddings = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)

        self.emb_layer_norm = nn.LayerNorm(self.config.hidden_size,eps=1e-12)
        self.emb_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.encoders = Encoder(self.config)
        self.pooler = nn.Sequential(
            OrderedDict(
                [
                    ("dense", nn.Linear(self.config.hidden_size, self.config.hidden_size)),
                    ("activation", nn.Tanh())
                ]
            )
        )

    def forward(self, input_ids, position_ids, segment_ids, encoder_mask):

        input_emb = self.word_embeddings(input_ids)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        seg_emb = self.segment_embeddings(segment_ids)
        if position_ids == None:
            position_ids = torch.arange(self.config.seq_len, dtype=torch.long,device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size())

        pos_emb = self.position_embeddings(position_ids)
        input_emb += pos_emb + seg_emb

        input_emb = self.emb_layer_norm(input_emb)
        input_emb = self.emb_dropout(input_emb)

        seq_out = self.encoders(input_emb, encoder_mask)

        pooler_out = self.pooler(seq_out[:,0:1,:])  # [cls] token embedding
        pooler_out=torch.squeeze(pooler_out)
        return pooler_out, seq_out

    def get_vocab_embedding(self):
        return self.word_embeddings.weight






