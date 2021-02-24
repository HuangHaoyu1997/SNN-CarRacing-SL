# -*- coding: utf-8 -*-
"""
读取监督学习模型，应用于CarRacing游戏，测试得分表现

Created on 2021年2月24日00:26
@author: hyhuang
Python 3.6.8
"""

from __future__ import print_function
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import time
from model import *
import argparse

gym.logger.set_level(40)

pretrain_model = torch.load('./checkpoint/cnn-test/cnn-test.pth')
# print(pretrain_model.keys()) # ['net', 'acc', 'epoch', 'acc_record', 'loss_test_record', 'loss_train_record', 'acc_label']
net = ConvNet(input_channel=3,output_channel=4) # .to(device)
net.load_state_dict(pretrain_model['net']) # 载入预训练模型
net.eval() # 设置为推理模式，避免单样本送进网络对batchnorm产生影响

env = gym.make('CarRacing-v0')

def action_projection(action):
    '''
    将离散动作映射成实数动作向量
    '''
    if action == 0:
        a = np.array([-0.3,0,0])
    elif action == 1:
        a = np.array([0.3,0,0])
    elif action == 2:
        a = np.array([0,0.5,0])
    elif action == 3:
        a = np.array([0.,0.,0.])
    return a

def startup():
    '''
    每局游戏开始之初，执行10~20次加速操作，此后再不加速
    '''
    random = np.random.randint(10,20)
    epi_r = 0
    for i in range(random):
        st,rt,done,_ = env.step(action_projection(2))
        epi_r += rt
    return st, epi_r, done, random

num_epi = 10
for i in range(num_epi):
    st = env.reset()
    done = False
    st, rt, done, random = startup()
    epi_r = rt
    action_count = [0,0,random,0]
    while done is not True:
        st = torch.tensor(st.copy()).reshape(3,96,96).unsqueeze_(0).float()/255
        action = torch.argmax(net(st)).item()
        if action == 2:
            action = 3
        action_count[action] += 1
        st, rt, done, _ = env.step(action_projection(action))
        epi_r += rt
        env.render()
    print(epi_r,action_count)


env.close()