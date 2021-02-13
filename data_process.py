import numpy as np
import os
import sys

action_list = os.listdir('./CarRacing/action')
reward_list = os.listdir('./CarRacing/reward')
state_list = os.listdir('./CarRacing/state')
'''
s0 = np.load('./CarRacing/0_s.npy')
s1 = np.load('./CarRacing/1_s.npy')
s2 = np.concatenate((s0,s1))
print(s2.shape)

a0 = np.load('./CarRacing/0_a.npy')
a1 = np.load('./CarRacing/1_a.npy')
a2 = np.concatenate((a0,a1))
print(a2.shape)
'''
aa = np.load('./CarRacing/action/'+action_list[0])
ss = np.load('./CarRacing/state/'+state_list[0])
rr = np.load('./CarRacing/reward/'+reward_list[0])
for i in range(1,len(action_list)):
    # print(i)
    a_tmp = np.load('./CarRacing/action/'+action_list[i])
    s_tmp = np.load('./CarRacing/state/'+state_list[i])
    r_tmp = np.load('./CarRacing/reward/'+reward_list[i])

    aa = np.concatenate((aa,a_tmp))
    rr = np.concatenate((rr,r_tmp))
    ss = np.concatenate((ss,s_tmp))

np.save('./CarRacing/aa.npy',aa)
np.save('./CarRacing/ss.npy',ss)
np.save('./CarRacing/rr.npy',rr)
# print(aa.shape,ss.reshape(-1,3,96,96).shape,rr.shape)
# print(aa.shape,a_tmp.shape)
# print(a_tmp.shape,s_tmp.shape,r_tmp.shape)

'''拼接新旧数据
aa = os.listdir('./data/action')
aa.sort()
rr = os.listdir('./data/reward')
rr.sort()
ss = os.listdir('./data/state')
ss.sort()

aaa = np.load('./data/aa.npy')
rrr = np.load('./data/rr.npy')
sss = np.load('./data/ss.npy')

print(aaa.shape)
print(rrr.shape)
print(sss.shape)

for i in range(len(aa)):
    a_tmp = np.load('./data/action/'+aa[i])
    r_tmp = np.load('./data/reward/'+rr[i])
    s_tmp = np.load('./data/state/'+ss[i])

    aaa = np.concatenate((aaa,a_tmp))
    rrr = np.concatenate((rrr,r_tmp))
    sss = np.concatenate((sss,s_tmp))
    print(sss.shape,aaa.shape,aaa.shape)

np.save('./data/aa.npy',aaa)
np.save('./data/rr.npy',rrr)
np.save('./data/ss.npy',sss)

'''

