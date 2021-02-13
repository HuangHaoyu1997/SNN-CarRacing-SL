# -*- coding: utf-8 -*-
"""
测试CNN在CarRacing数据集上的性能，state-->discrete action classification
Created on 2021年1月24日14:15
@author: hyhuang
Python 3.8.5
"""

from __future__ import print_function
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import time
from model import *
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
batch_size = 64
num_epochs = 40
learning_rate = 1e-4

names = 'Spike-CNN'
data_path =  './SNN/data/' #todo: input your data path
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_dataset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=True,  transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def cal_pixel_mean():
    R = torch.zeros((32,32))
    G = torch.zeros((32,32))
    B = torch.zeros((32,32))
    for i in range(len(train_dataset)):
        R += train_dataset[i][0][0]
        G += train_dataset[i][0][1]
        B += train_dataset[i][0][2]
    R = R/len(train_dataset)
    G = G/len(train_dataset)
    B = B/len(train_dataset)
    
    R_var = torch.zeros((32,32))
    G_var = torch.zeros((32,32))
    B_var = torch.zeros((32,32))
    for i in range(len(train_dataset)):
        R_var += (train_dataset[i][0][0]-R)**2
        G_var += (train_dataset[i][0][1]-G)**2
        B_var += (train_dataset[i][0][2]-B)**2
    R_var = R_var.mean().sqrt()
    G_var = G_var.mean().sqrt()
    B_var = B_var.mean().sqrt()
    return R,G,B,R_var,G_var,B_var

# test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# R,G,B,R_var,G_var,B_var = cal_pixel_mean() 
# print(R,G,B,R_var,G_var,B_var)

best_acc = 0  # best test accuracy
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

snn = SCNN(cfg_cnn,cfg_kernel,cfg_fc,batch_size,device=device)
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time() # 开始时间
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        '''
        images[:,0,:,:] = (images[:,0,:,:]-R)/R_var
        images[:,1,:,:] = (images[:,1,:,:]-G)/G_var
        images[:,2,:,:] = (images[:,2,:,:]-B)/B_var
        # print(images[0])
        '''
        outputs = snn(images) # F.log_softmax(snn(images),-1)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        # loss = -(outputs*labels_.to(device)).mean()
        loss = criterion(outputs, labels_.to(device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
            running_loss = 0
            # print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 30)
    # testing 
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs) # F.log_softmax(snn(inputs),-1)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = -(outputs.cpu()*labels_).sum()
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx % int(10000/batch_size) == 0:
                acc = 100. * float(correct) / float(total)
                # print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Epoch: %d,Testing acc:%.3f'%(epoch+1,100*correct/total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch % 5 == 0:
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './SNN/checkpoint/ckpt_' + names + '.t7')
        best_acc = acc
