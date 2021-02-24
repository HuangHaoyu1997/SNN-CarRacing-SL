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


pretrain_model = torch.load('./checkpoint/cnn-test/cnn-test.pth')
# print(pretrain_model.keys()) # ['net', 'acc', 'epoch', 'acc_record', 'loss_test_record', 'loss_train_record', 'acc_label']

net = ConvNet(input_channel=3,output_channel=4) # .to(device)
net.load_state_dict(pretrain_model['net'])

env = gym.make('CarRacing-v0')
st = env.reset()
print(st[0,0])
env.close()

