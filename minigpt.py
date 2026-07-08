from dataclasses import dataclass
from torch import nn
import torch
import math
import torch.nn.functional as F

@dataclass
class GPTconfig:
    vocab_size : int = 50257
    block_size : int = 1024
    n_layer    : int =  12
    n_embd     : int = 768
    n_head     : int = 12

config = GPTconfig()

#input Stage
class InputStage(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size,config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
    def forward(self, ids):
        seq_len = ids.size(-1)
        positions = torch.tensor(range(seq_len))
        tok =self.wte(ids)
        pos = self.wpe(positions)
        x = tok + pos
        return x

#Attention Mechanism
class Attention(nn.Module):
    def __init__(self,config):
     super().__init__()
     self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
     self.n_embd = config.n_embd
     self.n_head = config.n_head
     self.headdim = config.n_embd // config.n_head
     self.projection = nn.Linear(config.n_embd,config.n_embd)

    def forward(self, x):
      qkv = self.qkv(x)
      q,k,v = qkv.chunk(3, dim=-1)
      seq_len = x.size(0)
      q = q.reshape([seq_len, self.n_head, self.headdim])
      k = k.reshape([seq_len, self.n_head, self.headdim])
      v = v.reshape([seq_len, self.n_head, self.headdim])
      q = q.transpose(1,0)
      k = k.transpose(1,0)
      v = v.transpose(1,0)
      score = q @ k.transpose(1,2)
      score = score / math.sqrt(self.headdim)
      one_vector = torch.ones(seq_len,seq_len)
      mask_vector = torch.tril(one_vector)
      score = score.masked_fill(mask_vector == 0, float('-inf'))
      score = torch.softmax(score,dim=-1)
      output = score @ v
      output = output.transpose(0,1)
      output = output.reshape(seq_len, self.n_embd)
      output = self.projection(output)
      return output
       
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.w_up = nn.Linear (config.n_embd, 4 * config.n_embd)
        self.w_down = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        x = self.w_up(x)
        x = F.gelu(x, approximate='tanh')
        output = self.w_down(x)
        return output



    



      

      

