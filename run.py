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


pretrain_model = torch.load('./checkpoint/ckpt_cnn.pth')
print(pretrain_model['net'])
print(pretrain_model['acc'])
print(pretrain_model['epoch'])
print(pretrain_model['acc_record'])
print(pretrain_model.keys())