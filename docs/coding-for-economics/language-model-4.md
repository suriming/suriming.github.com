---
layout: default
title: Language model 4/6
parent: Coding For Economics
nav_order: 13
---

**final layer** 를 만들어보자.

마지막에 27 개의 글자를 predict 하고싶기 때문에 다음과 같이 만들어준다.

```python
W2 = torch.randn((100, 27))
b2 = torch.randn((27))
```

논문에서는 단어를 맞추려고 하기 때문에 사실 원 논문에서는 마지막에 17,000개로 분류한다.
그래서 그 부분에서 시간이 상당히 소요된다고 한다.

```python
logits = h@W2 + b2
```

우리는 이 logit 이 확률과 같은 역할을 하고 싶게 하기 때문에 다음과 같이 처리해준다.

```python
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
prob.shape # (25, 27)
```

negative log likelihood 를 계산할 수 있다.

```python
-prob[torch.arange(25), Y].log().mean()
```
