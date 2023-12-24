---
layout: default
title: Language model 5/6
parent: Coding For Economics
nav_order: 14
---

지금까지 만든 파라미터를 모아주고 갯수를 확인해보자.

```python
parameters = [C, W1, b1, W2, b2]

sum(p.nelement() for p in parameters)
```

3481 이 출력되어야한다.

## Cross Entropy

파이토치의 cross entropy 를 사용해서 가능한 여러가지 오류를 방지해서 loss 를 만들 수 있다.

```python
F.cross_entropy(logit, Y)
```
