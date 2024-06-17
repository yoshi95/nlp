import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # number of independent sentences to process in parallel
block_size = 256 # what is the max context length for predictions

max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# estimate loss every eval_iters
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -------------

torch.manual_seed(1337)

# read the tiny-shakespere.txt dataset
with open('tiny-shakespere.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create vocab by looking at all the unique chars
chars = sorted(
    # ordered array, based on some random ordering
    list(
        # creat a set of unique chars in dataset
        set(text)
    )
)
# vocab size is char size
vocab_size = len(chars)

# create tokens based on simple mapping dict

# string to int map
stoi = { ch:i for i, ch in enumerate(chars) }

# int to string map
itos = { i:ch for i, ch in enumerate(chars) }

# encoder: take string s, output array of int
encode = lambda s: [ stoi[c] for c in s ]

# decoder: take list of int, output string
decode = lambda l: ''.join([ itos[i] for i in l ])

data = torch.tensor( encode(text), dtype=torch.long)

# do train/validation split, 90/10
n = int(0.9 * len(data)) # first 90% will be used for training
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    '''
    get random batches of `batch_size` from train/validation data

    args:
        split: `train` or `val` data
    '''
    # use `train` dataset if split is `train`
    data = train_data if split == 'train' else val_data

    # get 4 batch of random indices
    ix = torch.randint(
        len(data) - block_size, # rand range [0, data_size - block_size]
        (batch_size, ) # shape: [batch_size, 1]
    )
    # stack 4 batches of 8 tokens into [4, 8] matrix
    # this is the context matrix
    x = torch.stack(
        [
            data[i:i+block_size] for i in ix
        ]
    )
    # stack 4 batches of `target` chars,
    # which is just 1 shifted over, since we are predicting next char
    y = torch.stack(
        [
            data[i+1 : i+block_size+1] for i in ix
        ]
    )

    # if GPU we need to move data to GPU
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # set to `eval/inference` mode
    # useful if model has BatchNorm and other inference features
    model.eval()
    for split in ['train', 'val']:
        # eval a batch of losses of size eval_iters.
        # get avg instead of keeping every loss because it could be noisy 
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    # set back to `train` mode
    model.train()
    return out

# very similar to BatchNorm1d
# BatchNorm makesure column dim (batch dim) is normal distribution
# LayerNorm make sure row dim is normal distribution
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        # self.momentum = momentum
        # self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with running momentum update)
        # self.running_mean = torch.zeros(dim)
        # self.running_var = torch.ones(dim)

    def __call__(self, x):
        
        # forward pass
        # we no longer need to check if we r in training/inference mode
        # if self.training:
        
        # change from 1 to 0 for BatchNorm
        xmean = x.mean(1,keepdim=True)
        xvar = x.var(1, keepdim=True)
        # else:
        #     xmean = self.running_mean
        #     xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta

        # the following is no longer needed for LayerNorm, because we are not calculating
        # statistics across sample (batch) dimensions
        
        # update buffers
        # if self.training:
        #     with torch.no_grad():
        #         self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        #         self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]



class Head(nn.Module):
    '''one head of self-attention'''
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute self attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,16) @ (B, 16, T) --> (B,T,T)
        # using mask, so this is a decoder
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # perform the weight aggration
        v = self.value(x) # (B,T,C)
        # matrix mult = aggregation, over value V
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    '''multiple heads of self-attention in parallel'''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([ Head(head_size) for _ in range(num_heads) ])
        # projection to residual pathway
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat( [h(x) for h in self.heads], dim=-1)
        out = self.dropout(
            self.proj(out)
        )
        return out

class FeedForward(nn.Module):
    '''
    a simple linear layer with non-linear forward
    '''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # 4 multiplyer from Attention is all You Need paper
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # same projection to residual pathway
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''
    transformer block: communication followed by comuptation
    - communication: attention mechanism between tokens
    - computation: feedfoward
    '''
    def __init__(self, n_embd, n_head):
        # n_emd: embedding dimension, n_head: the number of heads we like
        super().__init__()
        # embd=32, n_head=4, so head_size is 8
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # 2 layer norm layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # add x to itself, called residule/skip connection
        # this helps back propagation optimization issues arises when neural network gets deep.
        # the skip connection allows gradient to flow directly to input.
        # remember from minigrad addition distributes gradient equally between the 2 paths
        # see https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
        x = x + self.sa(
            # apply layer norm 1 to input
            # this is deviation from original paper
            # called pre-norm formulation
            self.ln1(x)
        )
        x = x + self.ffwd(
            self.ln2(x)
        )
        return x

# ------- MODEL --------------
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # this is now just token embedding, no longer probabilities of token 'a' followed by token 'b'
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # add layer to learn position info
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            # Block(n_embd, n_head=4),
            # Block(n_embd, n_head=4),
            # Block(n_embd, n_head=4),
            # nn.LayerNorm(n_embd)
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # token embedding -> logits, we need linear layer to learn the probabilities of next tokens embd
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # B = batch size
        # T = time dimension (each batch has blockSize/context size=8 words)
        # C=channel=vocab size (1 hot encoding?)
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B,T,n_emb)
        pos_emb = self.position_embedding_table(
            # indices from 0-T_1
            torch.arange(T, device=device)
        ) # (T,C)
        # combine token emb with position emb: (B,T,C) + (T,C), right aligned, broadcast applies
        x = tok_emb + pos_emb # now x has token info, as well as position info
        x = self.blocks(x)
        x = self.ln_f(x)
        # then pass it to focus on each individual head
        logits = self.ln_head(x) # (B,T,vocab_si)

        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            # we need to change the shape of the logits to match pytorch's cross_entropy method
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current vocab
        for _ in range(max_new_tokens):
            # crop idx to block size, because it cannot be bigger than attention size
            idx_cond = idx[:, -block_size:]
            # get the predictions, notice we dont pass in targets, loss=None in this case
            # prediction shape (B, T, C)
            logits, loss = self(idx_cond)
            # look at the last timestep only
            logits = logits[:, -1, :] # becoms (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            '''
            notice that we pass in the entire idx to the next `generate` call
            - this doesnt makes sense cuz it s a bigram model, which only looks at the prev char (hist size=1)
            - but this method is written in a generic manner, so we can expand to look back at longer history
            '''
            idx = torch.cat( (idx, idx_next), dim=1) # (B, T+1)

        return idx

model = BigramLanguageModel()
# same if GPU, we need to move model parameters to GPU
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ----- TRAIN ---------
for iter in range(max_iters):

    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # forward pass, evaluate the loss
    logits, loss = m(xb, yb)
    # reset all the gradients
    optimizer.zero_grad(set_to_none=True)
    # backward pass
    loss.backward()
    # update
    optimizer.step()

context = torch.zeros( (1,1), dtype=torch.long, device=device)
print(
    decode(
        # generate 500 tokens, start with '.', (B=1,T=1), 1 batch of 1 char='.'
        # returns (1, 101)
        m.generate(context, max_new_tokens=500)[0].tolist()
    )
)





