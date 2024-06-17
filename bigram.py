import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # number of independent sentences to process in parallel
block_size = 8 # what is the max context length for predictions

max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# estimate loss every eval_iters
eval_iters = 200
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

# ------- MODEL --------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # each token directly reads of logit for the next token from lookup table
        # nn.Embedding is a thin wrappr of tensor size [vocab_size, vocab_size]
        # or square matrix.
        # the weights/values for each Embedding dimension will be `learned`.
        # the learned values can be interpret as the probability of a char following the prev
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # B = batch size
        # T = time dimension (each batch has blockSize/context size=8 words)
        # C=channel=vocab size (1 hot encoding?)
        
        # idx and targets are both [batch, time] tensor of int
        # idx = inputs of size [batch, time], same as `xb` above
        # target = targe of size [batch, time], same as `yb` above 
        logits = self.token_embedding_table(idx) # (B,T,C)

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
            # get the predictions, notice we dont pass in targets, loss=None in this case
            # prediction shape (B, T, C)
            logits, loss = self(idx)
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

model = BigramLanguageModel(vocab_size)
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





