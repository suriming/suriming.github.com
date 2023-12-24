---
layout: default
title: Language model 6/6
parent: Coding For Economics
nav_order: 15
---

## Training

gradient 계산을 하도록 설정한다.

```python
for p in parameters:
    p.requires_grad = True
```

backpropagation 과 파라미터의 업데이트, 그라디언트 초기화를 잘 세팅해준다.

```python
for i in range(10):
    emb = C[X]
    h = torch.tanh(emb.view(25, 6) @ W1 + b1) # (25, 100)
    logit = h @ W2 + b2 # 25.27
    loss = F.cross_entropy(logit, Y)

    print(loss)

    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad
```

로스가 줄어드는 모습을 볼 수 있을 것이다.
