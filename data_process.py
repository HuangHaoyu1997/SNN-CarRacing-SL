# -*- coding: utf-8 -*-
'''
读入单个episode的数据，拼接新旧数据集
Created on 2021年2月13日22:51:45
@author: hyhuang
Python 3.6.8
'''
import numpy as np
import os
import sys

dir = './data/'

action_list = os.listdir(dir+'action')
reward_list = os.listdir(dir+'reward')
state_list = os.listdir(dir+'state')
# print(action_list,'\n',reward_list,'\n',state_list)

def concat_all():
    '''
    将若干个记录单一episode的文件拼成一个大文件
    concat all the single-episode files to a whole file 
    '''
    aa,rr,ss = [],[],[]
    for i in range(len(action_list)):
        if len(aa) == 0:
            aa = np.load(dir+'action/'+action_list[0])
            rr = np.load(dir+'reward/'+reward_list[0])
            ss = np.load(dir+'state/'+state_list[0])
        else:
            a_tmp = np.load(dir+'action/'+action_list[i])
            r_tmp = np.load(dir+'reward/'+reward_list[i])
            s_tmp = np.load(dir+'state/'+state_list[i])
            aa = np.concatenate((aa,a_tmp))
            rr = np.concatenate((rr,r_tmp))
            ss = np.concatenate((ss,s_tmp))
        print(aa.shape,rr.shape,ss.shape)
        return aa,rr,ss

def concat_one(aa,rr,ss):
    '''
    concat all the single-episode files to a whole file 
    '''
    for i in range(len(action_list)):
        if len(aa) == 0:
            aa = np.load(dir+'action/'+action_list[0])
            rr = np.load(dir+'reward/'+reward_list[0])
            ss = np.load(dir+'state/'+state_list[0])
        else:
            a_tmp = np.load(dir+'action/'+action_list[i])
            r_tmp = np.load(dir+'reward/'+reward_list[i])
            s_tmp = np.load(dir+'state/'+state_list[i])
            aa = np.concatenate((aa,a_tmp))
            rr = np.concatenate((rr,r_tmp))
            ss = np.concatenate((ss,s_tmp))
        print(aa.shape,rr.shape,ss.shape)
        return aa,rr,ss
def save_file():
    np.save(dir+'aa.npy',aa)
    np.save(dir+'ss.npy',ss)
    np.save(dir+'rr.npy',rr)

# aa,rr,ss = concat_all()
# save_file()

# 根据样本数量设定随机采样概率
'''
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader
numDataPoints = 10
data_dim = 5
bs = 2

# Create dummy data with class imbalance 9 to 1
data = torch.FloatTensor(numDataPoints, data_dim)
target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
                    np.ones(int(numDataPoints * 0.1), dtype=np.int32)))

print ('target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))

class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
print(target)
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in target])
print(samples_weight)
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
target = torch.from_numpy(target).long()
train_dataset = torch.utils.data.TensorDataset(data, target)

train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=0, sampler=sampler)
for index in BatchSampler(sampler, 10, True):
    print(index)
for i, (data, target) in enumerate(train_loader):
    print ("batch index {}, 0/1: {}/{}".format(
        i,
        len(np.where(target.numpy() == 0)[0]),
        len(np.where(target.numpy() == 1)[0])))

'''