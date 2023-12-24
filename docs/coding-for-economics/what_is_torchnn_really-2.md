---
layout: default
title: What is torch.nn really? 2/3
parent: Coding For Economics
nav_order: 6
---

# What is torch.nn really? 2/3

---

## Training

우선 데이터 몇개를 랜덤하게 만들어서 테스트해보자.

```python
torch.randint(10, (1,))
```

원래 epoch 은 주어진 데이터를 다 한번 돌면 1번이다.
그런데 다 한꺼번에 돌기에 메모리가 부족하기 때문에 보통 batch_size를 정해준다.

```
epochs = 1000
batch_size = 128
lr = 0.01

for epoch in range(epochs):
    idx = torch.randint(len(x_train), (batch_size,)) # 배치 사이즈만큼 랜덤하게 뽑기
    xs = x_train[idx]
    ys = y_train[idx]

    prob = model_1(xs) # 예측값
    loss = loss_func(prob, ys)

    if epoch % 100 == 0:
        print(loss)

    loss.backward()

    with torch.no_grad():
        weights -= weights.grad * lr
        bias -= bias.grad * lr

        weights.grad.zero_()
        bias.grad.zero_()

```

weight 값과 bias 를 업데이트 해주는 부분은 모형하고 상관 없는 부분이기 때문에
`with torch.no_grad():` 를 사용하여 gradient 를 계산하지 않는다.

실행시키면 loss 값이 점점 줄어드는 것을 볼 수 있다.

accuracy도 출력해보자.

```
print(loss_func(model_1(x_test), y_test))
print(accuracy(model_1(x_test), y_test))
```

## Using torch.nn.functional

앞에서 gradient 를 직접 업데이트를 수동으로 해줬는데,  
 `torch.nn.functional` 을 사용하면 그렇게 하지 않아도 된다!

loss functinon 도 직접 만들지 않고  
 `functional`에 포함된 `cross_entropy` 등을 사용할 수 있다.

```python
import torch.nn.functional as F

loss_func = F.cross_entropy

def model_2(inputs):
    return inputs @ weights + bias
```

앞에서 사용한 대로 코드를 다시 작성해주자.

```python
weights = torch.randn((784,10))
weights.requires_grad_()
bias = torch.zeros((10,), requires_grad=True)
```

```python
batch_size = 128

lr = 0.01
epochs = 1000

for epoch in range(epochs):

    idx = torch.randint(len(x_train), (batch_size,))
    xs = x_train[idx]
    ys = y_train[idx]

    prob = model_1(xs)
    loss = loss_func(prob, ys)
    if epoch % 100 == 0:
        print(loss)
    loss.backward()

    with torch.no_grad():
        weights -= weights.grad * lr
        bias -= bias.grad * lr

        weights.grad.zero_()
        bias.grad.zero_()
```

역시나 loss 가 줄어드는 것을 확인할 수 있고, accuracy를 출력해서 확인할 수 있다.

## Refactor using nn.Module

보통은 모델을 우리가 앞에서 했던 것 처럼 function 으로 만드는 것이 아니라 클래스로 만들게 된다.

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        return x @ self.weights + self.bias
```

initialize 하면서 파라미터를 등록해주는데 `requires_grad`같은 처리를 하지 않는 것을 볼 수 있다.

파라미터는 당연히 gradient 가 필요하기 때문에 `nn.Parameter`를 사용하면 이같은 처리를 자동으로 해주는 것이다.

모델을 만들고 데이터를 넣어 확인해볼 수 있다.

```python
model = MyModel()
model(x_train)
```

이제 학습을 시켜보자.

```python
batch_size = 128

lr = 0.01
epochs = 1000

for epoch in range(epochs):

    idx = torch.randint(len(x_train), (batch_size,))
    xs = x_train[idx]
    ys = y_train[idx]

    prob = model(xs)
    loss = loss_func(prob, ys)
    if epoch % 100 == 0:
        print(loss)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= p.grad * lr
        model.zero_grad()
```

파라미터로 지정해놓은 것을 돌면서 update 를 할 수 있다.

## Refactor using nn.Linear

파이토치에는 우리가 만들었던 linear 레이어도 이미 만들어져 있다.  
이번엔 그걸 사용해보자.

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
```

파라미터를 따로 적어주지 않아도 된다!
앞선 모델과 똑같이 학습을 시키고 테스트할 수 있다.

```python
batch_size = 128

lr = 0.01
epochs = 1000

for epoch in range(epochs):

    idx = torch.randint(len(x_train), (batch_size,))
    xs = x_train[idx]
    ys = y_train[idx]

    prob = model(xs)
    loss = loss_func(prob, ys)
    if epoch % 100 == 0:
        print(loss)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= p.grad * lr
        model.zero_grad()

print(loss_func(model(x_test), y_test))
print(accuracy(model(x_test), y_test))
```

## Refactor using torch.optim

더 쉽게 학습 할 수 있는 방법도 있다. optim 을 import 하자.

```python
from torch import optim
```

optimizer 을 정해줄 수 있다. adam 옵티마이저를 사용해보자.

```python
model = MyModel()
optimizer = optim.Adam(params=model.parameters())
```

learning rate 를 정해주지 않아도 default 로 들어가있는 게 있다.
이제 학습코드를 보자.

```python
batch_size = 128

lr = 0.01
epochs = 1000
for epoch in range(epochs):

        idx = torch.randint(len(x_train), (batch_size,))
        xs = x_train[idx]
        ys = y_train[idx]

        prob = model(xs)
        loss = loss_func(prob, ys)
        if epoch % 100 == 0:
            print(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

`optimizer.step()`과 `optimizer.zero_grad()`를 이용해서 업데이트를 해준 것을 볼 수 있다.

일반적인 모델은 보통 이러한 형태로 만들게 된다.
