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
        self.c_proj.NANOGPT_SCALING_INIT = 1 
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
        
        # ------------------- ORIGINAL START (mem access inefficient) ---------------------
        # attention (materialzes the large (T,T) matrix for all queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # ------------------- ORIGINAL END

        # ------------------- FLASH attention start (mem accesss efficient) ---------------------
        # https://arxiv.org/pdf/2307.08691
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # ------------------- FLASH attentsion end ----------------------------------------------


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
        self.c_proj.NANOGPT_SCALING_INIT = 1 

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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params for all submodules in transformer
        self.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALING_INIT'):
                std *= (2 * self.config.n_layer)**-0.5
            # initialize weights with normal dist std=0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # initialize to 0 for bias
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize weights with normal dist std=0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
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
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                # cross entropy doesnt like multi dimensional inputs
                # so flatten to 2d inputs
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

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

# get a data batch
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store in mem
        # encoder for gpt2
        enc = tiktoken.get_encoding('gpt2')
        # tiny shakespear dataset
        with open('tiny-shakespere.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T 
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets/labels
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y

import time

# auto detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # apple cpu (M1, M2 etc)
    device = 'mps'
print(f'using device: {device}')
# device = 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# train_loader = DataLoaderLite(B=16, T=1024)
train_loader = DataLoaderLite(B=8, T=1024)

# use tensorfloat32 bit precision
# result in less precision, but faster processing 
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304)) # add fake vocab so it s nice power of 2
model.to(device)
model = torch.compile(model)

# initial loss should be around -ln(vocab_size)=-ln(50257)=-10.28
# because all the weights are initialize random noise, so all vocabs have equal probability
# loss is -log likelihood
# print(loss) 

# use Adam optimizer, specifically AdamW, which allows for faster optimization than SGD
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# train 50 steps
for i in range(50):
    t0 = time.time()
    # -------------- GPU instructions --------------
    x, y = train_loader.next_batch()
    # tensor to() call returns a pointer of mem on device, so needs re-assignment
    x, y = x.to(device), y.to(device)

    # start 0 gradient
    optimizer.zero_grad()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)

    # now add gradients
    loss.backward()
    # update params
    optimizer.step()
    # -------------- GPU instructions end ----------

    # wait for all GPU instructions are done
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time diff in ms
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)

    # loss is a tensor of 1 on GPU, calling `item()` converts it to float on CPU
    print(f'step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}')

import sys; sys.exit(0)
