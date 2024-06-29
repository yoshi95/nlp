from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming tho
        self.register_buffer(
            'bias',
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh - 'number of heads'
        # hs - 'head size'
        # C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_heads=12, hs=64, so nh*hs=768 channels (n_embd) in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # attention (materialzes the large (T,T) matrix for all queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GeLU is like ReLU, but has smooth transition at 0
        # https://paperswithcode.com/method/gelu
        # the approximate version was created for perf, but that should no longer be the case,
        # and the need to use it
        # using approximage here to reproduced GPT-2
        #
        # Note: ReLU flat tail could cause dead neurons, because any value map to flat tail region will be 0
        # GeLU tail, because of smoothing, will always have some gradient, which works out better
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x) # linear
        x = self.gelu(x) # non-linear activation
        x = self.c_proj(x) # linear
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # this is a repeated map-reduce operation

        # add attention to itself
        # rememeber: attention is a communication mechanishm between 1024 tokens (block_size)
        # it is an aggregation/pooling/weighted-sum, a [array.reduce] operation
        # this is part of residual connection
        # this is where tokens communicate
        x = x + self.attn(self.ln_1(x))
        # multilayer percentron, feedfoward network to learn
        # MLP doesnt exchange info between tokens, it is a [array.map] operation
        # this is where tokens think by itself
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq length
    vocab_size: int = 50257 # number of tokens: 50k BPE merge + 256 bytes token + 1 <|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dim

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # replication hugging face transformer so we can load their GPT2 model as well
        # this way we can load huggingface model weights to prove that the setup is correct.
        # then us random weights and train
        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # hidden layers
            h = nn.ModuleList([ Block(config) for _ in range(config.n_layer) ]),
            # layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # linear layer head
        # projects hidden states from embedding space to vocab space
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        # idx is of shape (B,T), B=batch dim, T=time dim, i.e. sequence length
        B, T = idx.size()
        # make sure T=seq length is not bigger than block_size=max seq length
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size {self.config.block_size}'
        # foward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # note: pos_emb is the same for the batch, so there is broadcasting
        # forward the blocks of the trnasformer
        for block in self.transformer.h:
            x = block(x)
        # foward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B,T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        '''
        loads pretrained GPT-2 model weights from huggingface
        '''
        assert model_type in {'gpt2', 'gpt-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints, no matter the size
        config_args['block_size'] = 1024 # always 1024 for GPT model

        # create a from_scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # print(sd_keys)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore

        # these needs to be manually transposed
        # basically OpenAI checkpoints use a 'Conv1D' module, but we only want to use vanilla linear
        # this means that we have to transpos these weights when we import them
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight'
        ]
        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ------------------------------------------------------------------------------------------------

# auto detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # apple cpu (M1, M2 etc)
    device = 'mps'
print(f'using device: {device}')


# sample model 5 times, i.e. 5 different sequences
num_return_sequences = 5
max_length = 30 # up to 30 tokens each sequence

model = GPT.from_pretrained('gpt2')
model.eval() # set `eval` mode, vs `training` mode, there might be layer behavior diff during training
model.to(device)

# initialize prefix tokens for generation
import tiktoken
# encoder for gpt2
enc = tiktoken.get_encoding('gpt2')
# encode the string using BPE
tokens = enc.encode("Hello, I'm a languge model,")
# convert it to torch tensor
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# duplicate 5 times, because we want to sample the model 5 times using these prefix tokens
# B=5, T=8 
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5,8)
x = tokens.to(device)

# now generate x is (B,T), B=5, T=8
# set seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B,T,vocab_size)
        # take the logits at the end position
        # imagine plane of size (B,T), z-direction is vocab_size, [-1] is the last time dim token
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B,1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        # append the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)