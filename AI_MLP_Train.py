# coding: utf-8
import sys, os

sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
from AI_MLP_Three_Layernet import ThreeLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 드랍아웃 사용여부,
use_dropout = False
dropout_ratio = 0.2

#입력 뉴런 784개(이미지의 크기 28*28 = 784), 은닉층에 50개의 뉴런 배치, 출력층 10개(0~9)
network = ThreeLayerNet(input_size=784, hidden_size=50, output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)

#반복횟수 10000, 100개의 미니배치, 0.1의 학습율 설정
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
repeat_list = []

# 1에폭당 반복수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    repeat_list.append(i)

    #1에폭당 정확도(훈련, 검증) 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Train: %.4f, Test: %.4f" %(train_acc, test_acc))

plt.scatter(repeat_list,train_loss_list,s=1) #x축은 반복 횟수 y축은 손실 값
plt.show()

