---
layout: default
title: Tensorflow 2/2
parent: Coding For Economics
nav_order: 2
---

텐서플로우 튜토리얼 [Celcius_to_fahrenheit]({{"https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb"}}/) 을 진행하면서 코드를 살펴보고 Tensorflow 1/2 에서 배운 내용을 복습해보자.

**Enumerate의 사용**

Enumerate를 사용하면 순서와 값 자체를 동시에 얻을 수 있다. 이를 통해 Celsius와 Fahrenheit 간의 변환을 출력해볼 수 있다.

```python
for i, c in enumerate(celsius_q):
print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
```

**Feature, Label, Example**

Feature: 모델에 들어가는 입력

Label: 모델의 출력

Example: 입력과 출력의 쌍

**Dense Layer**

모든 노드로부터 입력을 받는 레이어.

두 번째 레이어부터는 어차피 모든 입력을 받기 때문에 인풋에 대한 추가적인 설정은 필요하지 않다.

**Optimizer**

어떤 옵티마이저를 사용하느냐에 따라 학습 속도와 효율이 달라진다.

**모델 학습**

model.fit을 사용하여 모델을 학습시킨다.
최소 세 가지 인자가 필요하다: 입력, 출력, 반복 횟수(epochs).

```python
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

```

{: .note }
이러한 epoch 은 hyperparameter 중 하나이다.  
하이퍼파라미터는 직접 조절해줄 수 있는 파라미터이다.  
학습할 때 -learning rate\*gradient 만큼 진행하게 되는데 이때 learning rate 도 하이퍼파라미터이다.

또한 위와 같이 학습 과정을 history에 저장해서 볼 수 있다.

**모델 평가와 예측**

```python
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
```

위와 같은 코드로 epoch가 진행되면서 loss 가 어떻게 되는지 시각화하여 모델의 성능을 평가할 수 있다.

**모델 예측**

```python
print(model.predict([100.0]))
```

모델이 스스로 찾아낸 예측값을 출력한다.
