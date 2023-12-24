---
layout: default
title: Bigram - Probability
parent: Coding For Economics
nav_order: 9
---

# Bigram - Probability

---

지난번 만든 tensor 로 row 를 봤을 때 각각 그 다음 문자가 나타날 확률을 구할 수 있다.

마지막 element 에 대해서 해보자.

```python
p = N[-1].float()
p = p / p.sum()
p
```

이렇게 만든 것으로 multinomial 분포를 만들고 이에 따라 item을 뽑을 수 있다.

```python
g = torch.Generator().manual_seed(1999)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
ix
```

이 matrix 전체를 확률로 만들자.

```python
P = (N+10000).float()
P /= P.sum(1, keepdims=True)

```

row 별로 더해야 하기 때문에 1 인덱스를 쪽으로 sum 한다.

또한 keepdims=True 를 통해 sum 을 해주어야 (27, 1) 의 shape 을 만들 수 있는데, 이렇게 해야 두번째 줄 코드를 실행할 수 있기 때문이다.

{: .note}
두번째 줄에서는 Broadcasting 이 일어난다.  
Broadcasting 이 일어나면 matrix 의 dimension 이 달라도 맞춰서 계산한다.

이 모형을 이용해서 predict를 해보자.

```python
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0: # 만약 period 가 예측되면 끝냄
            break
    print(''.join(out))
```

별로 사람이름같지 않은 결과가 만들어질 것이다.
bigram 모형의 한계인데, neural network 로 넘어가서 다른 모형을 만들어보자.

모델이 얼마나 퀄리티가 좋은지 에 대해 negative log likelihood 를 지표로 사용할 수 있다.

```python
logl = 0.0
n = 0

for w in words:
    w = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(w[:-1],w[1:]):
        id1 = stoi[ch1]
        id2 = stoi[ch2]
        prob = P[id1,id2]
        logprob = torch.log(prob)
        logl += logprob
        n += 1
        #print(f'{ch1}{ch2} {prob: .4f} {logprob: .4f}')

nll = -logl/n # 평균

print(f"{nll}")
```

negative log likelihood 이므로 우리는 이걸 작게 만들어야 한다!

{: .note}
log 를 취해야되는데 probability 가 0 이되는 부분을 어떻게 해결할까?  
위에서는 torch.log(prob)를 사용해 처리해줬다.  
다른 방법으로는 N에다가 base로 1을 더해줄 수도 있다.
