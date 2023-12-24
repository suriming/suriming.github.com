---
layout: default
title: Decoder - model
parent: Coding For Economics
nav_order: 19
---

사용할 모델을 먼저 살펴보자.

```python
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        self.pos_emb = nn.Embedding(block_size, n_emb)

        self.blocks = nn.Sequential(
            Block(n_emb, n_head=4),
            Block(n_emb, n_head=4),
            Block(n_emb, n_head=4),
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx) # (B,T,E)
        pos_emb = self.pos_emb(torch.arange(T, device=device)) # (T,E)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,C)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)

        return idx

```

먼저 initialize 부분을 살펴보자.

```python
self.tok_emb = nn.Embedding(vocab_size, n_emb)
self.pos_emb = nn.Embedding(block_size, n_emb)
```

토큰 임베딩은 vocab을 인자로 받아서 각 vocab에 따라 임베딩을 부여하는 것이고  
포지셔널 임베딩은 block 내에서 어느 위치에 있느냐에 따라서만 값을 다르게 하면 되는 것이기 때문에  
 `block_size`를 인자로 받고 있다.

```python
tok_emb = self.tok_emb(idx) # (B,T,E)
```

B는 batch, T는 length, E는 임베딩 디멘션이다. 이게 이 코드의 아웃풋으로 나온다.

이를 포지셔널 임베딩과 심플하게 더해서 들어갈 인풋이 만들어진다.

```python
pos_emb = self.pos_emb(torch.arange(T, device=device)) # (T,E)
x = tok_emb + pos_emb
```

즉 그림에서 포지셔널 임베딩이 끝난 바로 그 시점이다.

```python
self.blocks = nn.Sequential(
    Block(n_emb, n_head=4),
    Block(n_emb, n_head=4),
    Block(n_emb, n_head=4),
)
self.lm_head = nn.Linear(n_emb, vocab_size)
```

다음으로 그림에서 그 다음으로 나와있는 큰 블록들인데,  
원래 여러개를 반복하게 되어있다.  
원 논문에서는 6번 반복되어있다고 한다.

sequential 이므로 차례대로 통과한다.

그리고 그림에서처럼 마지막으로 linear layer 을 통과한다.

```python
logits = self.lm_head(x) # (B,T,C)

if targets == None:
    loss = None
else:
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
```

각각의 확률을 맞춰야하기 때문에 후에 이렇게 처리를 해준다.

Multi head attention 은 scaled dot product를 나눠서 계산하는 것과 비슷하다.

```python
class Block(nn.Module):

    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x
```

원논문에서는 multi head attention 을 해주고 normalize 하고있는데,  
후에 normalize 를 먼저 해주는 게 더 효과가 좋다는 연구 결과들이 나았다고 한다.  
이 코드에서도 먼저 add&normalize 를 해주고 있다.

feed forward는 linear layer 과 비슷하다고 보면 된다.

```python
class FeedForward(nn.Module):

    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
```
