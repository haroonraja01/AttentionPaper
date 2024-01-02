import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class BasicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear()

    def forward(self, x):
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # d_model = embedding dim = 512
        self.fc = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)
        #self.relu = F.relu()

    def forward(self, q, k, v):
        qk = self.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[2]))
        return torch.bmm(qk, v)


class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        qk = self.softmax(torch.bmm(q, k.transpose(1, 2))/math.sqrt(q.shape[2]))
        return torch.bmm(qk, v)

    #def multihead_attention(self):


