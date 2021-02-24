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
from car_racing import CarRacing
import argparse

gym.logger.set_level(40)
num_epi = 100
render = False
device = torch.device('cpu')
model = 'snn' # 'cnn'
model_path = './checkpoint/snn/snn.pth' # './checkpoint/CNN-CarRacing/CNN-CarRacing.pth'

pretrain_model = torch.load(model_path)
# print(pretrain_model.keys()) # ['net', 'acc', 'epoch', 'acc_record', 'loss_test_record', 'loss_train_record', 'acc_label']
if model == 'cnn':
    net = ConvNet(input_channel=3,output_channel=4) # .to(device)
if model == 'snn':
    net = SCNN(input_channel=3,output_channel=4,batch_size=128,device=device)
net.load_state_dict(pretrain_model['net']) # 载入预训练模型
net.eval() # 设置为推理模式，避免单样本送进网络对batchnorm产生影响

# env = gym.make('CarRacing-v0')
env = CarRacing()

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
    times = np.random.randint(15,16)
    epi_r = 0
    for i in range(times):
        st,rt,done,_ = env.step(action_projection(2))
        epi_r += rt
    return st, epi_r, done, times

def repeat():
    times = np.random.randint(3,5)
    epi_r = 0
    for i in range(times):
        st,rt,done,_ = env.step(action_projection(3))
        epi_r += rt
    return st, epi_r, done, times

r_list = []
for i in range(num_epi):
    st = env.reset()
    done = False
    st, epi_r, done, times = startup() # 加速3-5个timestep
    action_count = [0,0,times,0]

    while done is not True:
        if model == 'cnn': 
            st = torch.tensor(st.copy()).reshape(3,96,96).unsqueeze_(0).float()/255
            action = torch.argmax(net(st)).item()
        if model == 'snn': 
            st_ = torch.tensor(st.copy()).reshape(3,96,96).unsqueeze_(0).float()/255
            st = torch.zeros((128,3,96,96)).float()
            st[0] = st_
            action = torch.argmax(net(st)[0]).item()
        
        action_count[action] += 1 # 统计原始动作数量比例

        if action == 2: # 不再加速
            action = 3
        
        if action_count[0] >= 10 and action == 0: # 纠正连续左转
            st, rt, done, times = repeat()
            action_count[3] += times
            epi_r += rt
            action_count[0] = 0
            continue
        elif action_count[1] >= 10 and action == 1: # 纠正连续右转
            st, rt, done, times = repeat()
            action_count[3] += times
            epi_r += rt
            action_count[1] = 0
            continue
        
        st, rt, done, _ = env.step(action_projection(action))
        epi_r += rt
        if render: env.render()

    print(epi_r,action_count)
    r_list.append(epi_r)
env.close()
print('average episodic reward:',np.array(r_list).mean())

