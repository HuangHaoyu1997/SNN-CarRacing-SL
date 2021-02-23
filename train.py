# -*- coding: utf-8 -*-
"""
采用CNN对CarRacing游戏的人类交互数据集进行训练
输入state，输出是离散动作的分类概率，将RL控制问题转化为监督学习

Created on 2021年1月24日16:13
@author: hyhuang
Python 3.6.8
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler
import os
import time
from model import *
import argparse

parser = argparse.ArgumentParser(description='CNN CarRacing Supervised Learning')
parser.add_argument('--model', type=str, default='cnn' ,help='cnn model or snn model')
parser.add_argument('--epochs', type=int, default=100 ,help='training epoch')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--device', type=int,default=0,help='index of GPU device')
parser.add_argument('--ckpt_name', type=str, default='CNN-CarRacing', help='name of checkpoint file')

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

data_path =  './data/' 
device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
print(device)

def generate_data(data_path, test_ratio=0.2):
    '''
    aa:离散动作，0左转，1右转，2加速，3不动
    左转：[-0.3,0,0]
    右转：[0.3,0,0]
    加速：[0,0.5,0]
    不动：[0.,0.,0.]
    '''
    ss = torch.tensor(np.load(data_path+'sss_balance.npy')).reshape(-1,3,96,96).float()/255
    aa = torch.tensor(np.load(data_path+'aaa_balance.npy')).long()
    rr = torch.tensor(np.load(data_path+'rrr_balance.npy'))

    sample_index = random.sample(range(len(aa)),len(aa))
    test_num = int(len(aa)*test_ratio//args.batch_size)*args.batch_size # 按比例分割训练集和测试集
    ss_test = ss[sample_index[:test_num]]
    rr_test = rr[sample_index[:test_num]]
    aa_test = aa[sample_index[:test_num]]

    ss_train = ss[sample_index[test_num:]]
    rr_train = rr[sample_index[test_num:]]
    aa_train = aa[sample_index[test_num:]]

    return ss_train,aa_train,rr_train,ss_test,aa_test,rr_test

# 生成训练、测试数据集
ss_train,aa_train,rr_train,ss_test,aa_test,rr_test = generate_data(data_path)
train_num = len(aa_train)
test_num = len(aa_test)

# 计算各类样本之数量
number_0_train = len(torch.where(aa_train==0)[0])
number_1_train = len(torch.where(aa_train==1)[0])
number_2_train = len(torch.where(aa_train==2)[0])
number_3_train = len(torch.where(aa_train==3)[0])
print('num of labels in train set:',number_0_train,number_1_train,number_2_train,number_3_train,train_num)
number_0_test = len(torch.where(aa_test==0)[0])
number_1_test = len(torch.where(aa_test==1)[0])
number_2_test = len(torch.where(aa_test==2)[0])
number_3_test = len(torch.where(aa_test==3)[0])
print('num of labels in test set:',number_0_test,number_1_test,number_2_test,number_3_test,test_num)

# 按样本数量设置采样概率
class_sample_count = np.array([number_0_train,number_1_train,number_2_train,number_3_train])
weight = 1. / class_sample_count
samples_weight = np.array([weight[a] for a in aa_train])
samples_weight = torch.from_numpy(samples_weight).double()
weight_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


# train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
# train_dataset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=True,  transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
# test_set = torchvision.datasets.CIFAR10(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

if args.model == 'cnn':
    net = ConvNet(input_channel=3,output_channel=4).to(device)
if args.model == 'snn':
    net = SCNN(input_channel=3,output_channel=4,batch_size=args.batch_size,device=device).to(device)

if args.optim == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
elif args.optim == 'SGD':
    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr, momentum=0.9,weight_decay=1e-5)
criterion =torch.nn.MSELoss()

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

for epoch in range(args.epochs):
    running_loss,i = 0,0
    start_time = time.time() # 开始时间
    # for i, (images, labels) in enumerate(train_loader):
    # for index in BatchSampler(SubsetRandomSampler(range(train_num)), args.batch_size, True): # drop_last=True则表示最后不足一个batch的数据被丢弃
    for index in BatchSampler(weight_sampler,args.batch_size,True): # drop_last=True则表示最后不足一个batch的数据被丢弃
        net.zero_grad()
        optimizer.zero_grad()
        images = ss_train[index].to(device)
        outputs = net(images) 
        
        labels_ = torch.nn.functional.one_hot(aa_train[index], 4).to(device).float()
        # labels_ = torch.zeros(args.batch_size, 4).scatter_(1, aa_train[index].view(-1, 1), 1).to(device)
        # loss = -(outputs.log() * labels_).mean() # cross entropy
        loss = criterion(outputs, labels_.to(device))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%300 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'%(epoch+1, args.epochs, i+1, train_num//args.batch_size,running_loss ))
            running_loss = 0
            # print('Time elasped:', time.time()-start_time)
        i += 1
    correct = 0
    total = 0
    # optimizer = lr_scheduler(optimizer, epoch, learning_rate, 30)
    # testing 
    total_loss = 0
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(test_loader):
        for index in BatchSampler(SubsetRandomSampler(range(test_num)), args.batch_size, True):
            images = ss_test[index].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            labels_ = torch.nn.functional.one_hot(aa_test[index], 4).to(device).float()
            # labels_ = torch.zeros(args.batch_size, 4).scatter_(1, aa_test[index].view(-1, 1), 1)
            
            loss = criterion(outputs, labels_)
            # loss = -(outputs.log()*labels_).sum()
            total_loss += loss.item()
            
            _, predicted = outputs.cpu().max(1)
            correct += float(predicted.eq(aa_test[index].cpu()).sum().item())
    
    print('Epoch: %d,Testing acc:%.3f'%(epoch+1,100 * correct/test_num))
    acc = 100. * float(correct) / float(test_num)
    acc_record.append(acc)
    if epoch % 5 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + args.ckpt_name + '.t7')
        best_acc = acc

plt.plot(acc_record)
plt.xlabel('epoch')
plt.ylabel('testing acc')
plt.savefig('testing_acc_'+ args.ckpt_name +'.png')