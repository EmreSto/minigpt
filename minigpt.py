from dataclasses import dataclass
from torch import nn
import torch
import math
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


@dataclass
class GPTconfig:
    vocab_size : int = 50257
    block_size : int = 1024
    n_layer    : int =  12
    n_embd     : int = 768
    n_head     : int = 12

config = GPTconfig()

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

#MLP   
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

#One transformer Block    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self,x):
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
#Model
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, ids):
        seq_lens = ids.size(-1)
        positions = torch.tensor(range(seq_lens))
        tok = self.wte(ids)
        pos = self.wpe(positions)
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x =  self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# name translate function
def translate(hf_name):
    name = hf_name
    name = name.replace("transformer.", "")
    name = name.replace("h.", "blocks.")
    name = name.replace("ln_1", "ln1")
    name = name.replace("ln_2", "ln2")
    name = name.replace("attn.c_attn", "attention.qkv")
    name = name.replace("attn.c_proj", "attention.projection")
    name = name.replace("mlp.c_fc", "mlp.w_up")
    name = name.replace("mlp.c_proj", "mlp.w_down")
    return name
model = GPT(config)
hf = GPT2LMHeadModel.from_pretrained('gpt2')
my_sd = model.state_dict()
with torch.no_grad():
    for hf_name, hf_tensor in hf.state_dict().items():
        my_name = translate(hf_name)
        if my_name not in my_sd:
            continue
        if ('c_attn' in hf_name or 'c_proj'in hf_name or 'c_fc' in hf_name) and hf_name.endswith(".weight"):
            my_sd[my_name].copy_(hf_tensor.t())
        else:
            my_sd[my_name].copy_(hf_tensor)
ids = torch.tensor([15496, 11, 314, 716])
mine = model(ids)
theirs = hf(ids.unsqueeze(0)).logits.squeeze(0)
print(torch.allclose(mine, theirs, atol=1e-4))




      

      

