from dataclasses import dataclass
from torch import mode, nn
import torch
import math
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
import time


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

    def forward(self, x, cache = None):
      qkv = self.qkv(x)
      q,k,v = qkv.chunk(3, dim=-1)
      batch = x.size(0)
      seq_len = x.size(1)
      q = q.reshape([batch, seq_len, self.n_head, self.headdim])
      k = k.reshape([batch, seq_len, self.n_head, self.headdim])
      v = v.reshape([batch, seq_len, self.n_head, self.headdim])
      q = q.transpose(2,1)
      k = k.transpose(2,1)
      v = v.transpose(2,1)
      if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
      score = q @ k.transpose(2,3)
      score = score / math.sqrt(self.headdim)
      if cache is None:
        one_vector = torch.ones(seq_len,seq_len)
        mask_vector = torch.tril(one_vector)
        score = score.masked_fill(mask_vector == 0, float('-inf'))
      score = torch.softmax(score,dim=-1)
      output = score @ v
      output = output.transpose(1,2)
      output = output.reshape(batch ,seq_len, self.n_embd)
      output = self.projection(output)
      return output, (k, v)

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
    def forward(self,x, cache = None):
        attn_out, new_cache = self.attention(self.ln1(x),cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x , new_cache
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

ids = torch.tensor([15496, 11, 314, 716]).unsqueeze(0)
mine = model(ids)
theirs = hf(ids).logits
print(torch.allclose(mine, theirs, atol=1e-4))


def generate(model, ids, max_new_tokens,mode, T, k , p ):

    for _ in range(max_new_tokens):
        logits = model(ids)
        last_pos = logits[:, -1, :]
        temp_div = last_pos / T
        if mode == "greedy":
            ntid = torch.argmax(temp_div, dim=-1).unsqueeze(-1)
        elif mode == "top-k":
            top_k = torch.topk(temp_div, k=k, dim=-1)
            cutoff= top_k.values[:, -1:]
            filtered = temp_div.masked_fill(temp_div < cutoff, float('-inf'))
            softmax = torch.softmax(filtered, dim=-1)
            ntid = torch.multinomial(softmax, num_samples=1)
        elif mode == "top-p":
            prob = temp_div.softmax(dim=-1)
            sorted_prob = torch.sort(prob, descending = True)
            cumulative_prob = torch.cumsum(sorted_prob.values, dim=-1)
            sorted_indices_to_remove = cumulative_prob > p
            rolled = torch.roll(sorted_indices_to_remove, shifts=1, dims=-1)
            rolled[:,0] = False
            vocab_mask = torch.zeros_like(sorted_prob.indices, dtype= torch.bool).scatter(1, sorted_prob.indices, rolled)
            m_fill = temp_div.masked_fill(vocab_mask, float('-inf'))
            softmax = torch.softmax(m_fill,dim=-1)
            ntid = torch.multinomial(softmax,num_samples=1)
        ids = torch.cat([ids, ntid], dim=1)
    return ids

tokenizer = AutoTokenizer.from_pretrained('gpt2')

prompt = "The capital of France is "

inputs = tokenizer(prompt, return_tensors ='pt')

input_ids = inputs['input_ids']

time_1 = time.time()

output_ids = generate(model, input_ids,20 , mode= "greedy", k= 40, T=0.7, p =0.9)

time_2 = time.time()

new_tokens = output_ids[0][input_ids.shape[1]:]
generated_text = tokenizer.decode(new_tokens)
full_text = tokenizer.decode(output_ids[0])

elapsed_time = time_2-time_1

tokens_second = len(new_tokens)/elapsed_time

print(f"Generated text: {generated_text}")
print(f"Full text : {full_text}")
print(f"Elapsed time: {elapsed_time:.4f} seconds")
print(f"Tokens per second: {tokens_second:.2f} tokens/s")

