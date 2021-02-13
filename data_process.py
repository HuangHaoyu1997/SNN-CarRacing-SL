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
print(action_list,'\n',reward_list,'\n',state_list)

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

aa,rr,ss = concat_all()
save_file()