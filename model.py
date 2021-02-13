import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    input_channel: input image of CarRacing game state
    output_channel: output tensor shape, representing the prob. of each discrete action 
    """
    def __init__(self, input_channel,output_channel):
        super(ConvNet, self).__init__()
        self.cnn_base = nn.Sequential(                   # input shape (4, 96, 96)
            nn.Conv2d(input_channel, 8, kernel_size=4, stride=2),
            nn.BatchNorm2d(8),                           # using BN can accelerate the training process and improve performance slightly
            nn.ReLU(),  
            nn.Conv2d(8, 16, kernel_size=3, stride=2),   # (8, 47, 47)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # (64, 5, 5)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),# (128, 3, 3)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )                                                # output shape (256, 1, 1)
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.BatchNorm2d(100), nn.ReLU(), nn.Linear(100, output_channel))
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        x = torch.softmax(self.fc(x),-1)
        return x

class ActFun(torch.autograd.Function):
    '''
    Activation function for LIF model
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # 将Tensor转变为Variable保存到ctx中
        return input.gt(0.).float()  # input比0大返回True的float，即1.0，否则0.0

    @staticmethod
    def backward(ctx, grad_output, lens=0.5):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens # 膜电位处在发放阈值的附近就令导数=1
        return grad_input * temp.float()

# 使用apply方法对自己定义的激活函数取个别名
act_fun = ActFun.apply

def mem_update(ops, inputs, spike, mem, thr=0.3, decay=0.1, activation=None):
    '''
    update function of LIF SNN, similar with GRU model
    
    ops:    Linear operation or Conv operation
    inputs: input spike tensor from (L-1) layer at time t
    spike:  spike tensor fired by (L) layer at time t-1
    mem:    membrane potential of (L) Layer at time t-1
    thr:    firing threshold, learnable optional but few improvement
    decay:  like forgetting gate in LSTM, controling the information flow of membrane through time 
    
    '''
    state = ops(inputs)  # 当前状态,
    # mem是膜电位，spike=1即上一状态发放了脉冲，则mem减去一个发放阈值thr
    # spike=0则上一时刻未发放脉冲，则膜电位继续累积
    # Here we use a firing-and-substration strategy and formulize the learnable decay parameters
    
    mem = state + mem * (1 - spike) * decay# .clamp(min=0., max=1.)
    # mem = state + (mem - spike * thr) * decay# .clamp(min=0., max=1.)
    
    now_spike = act_fun(mem - thr)
    return mem, now_spike.float()

class SCNN(nn.Module):
    '''
    input_channel:  input image of CarRacing game state
    output_channel: output tensor shape, representing the prob. of each discrete action 
    batch_size：    batch size when training
    device:         GPU device or CPU
    '''
    def __init__(self,input_channel,output_channel,batch_size,device,time_window=6):
        super(SCNN, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.output_channel = output_channel
        self.time_window = time_window
        
        self.conv1 = nn.Conv2d(input_channel, 8, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        
        self.fc1 = nn.Linear(256,100)
        self.fc2 = nn.Linear(100, output_channel)

        # self.thr = nn.Parameter(torch.rand(output_channel, device=self.device))  # tensor变parameter，learnable
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(self.batch_size, 8, 47, 47, device=self.device) # 第1卷积层output feature map size=[batch,channel,width,height]
        c2_mem = c2_spike = torch.zeros(self.batch_size, 16, 23, 23, device=self.device)
        c3_mem = c3_spike = torch.zeros(self.batch_size, 32, 11, 11, device=self.device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, 64, 5, 5, device=self.device)
        c5_mem = c5_spike = torch.zeros(self.batch_size, 128, 3, 3, device=self.device)
        c6_mem = c6_spike = torch.zeros(self.batch_size, 256, 1, 1, device=self.device)
        
        fc1_mem = fc1_spike = fc1_sumspike = torch.zeros(self.batch_size, 100, device=self.device) 
        fc2_mem = fc2_spike = fc2_sumspike = torch.zeros(self.batch_size, self.output_channel, device=self.device)

        for step in range(self.time_window): # 仿真时间，即发放次数
            x = input > torch.rand(input.size(), device=self.device) # prob. firing

            c1_mem, c1_spike = mem_update(ops=self.conv1, inputs=x.float(), mem=c1_mem, spike=c1_spike)
            c2_mem, c2_spike = mem_update(ops=self.conv2, inputs=c1_spike, mem=c2_mem, spike=c2_spike)
            c3_mem, c3_spike = mem_update(ops=self.conv3, inputs=c2_spike, mem=c3_mem, spike=c3_spike)
            c4_mem, c4_spike = mem_update(ops=self.conv4, inputs=c3_spike, mem=c4_mem, spike=c4_spike)
            c5_mem, c5_spike = mem_update(ops=self.conv5, inputs=c4_spike, mem=c5_mem, spike=c5_spike)
            c6_mem, c6_spike = mem_update(ops=self.conv6, inputs=c5_spike, mem=c6_mem, spike=c6_spike)
            
            c6_flatten = c6_spike.view(self.batch_size, -1)
            fc1_mem, fc1_spike = mem_update(ops=self.fc1, inputs=c6_flatten, mem=fc1_mem, spike=fc1_spike)
            fc1_sumspike += fc1_spike
            fc2_mem, fc2_spike = mem_update(ops=self.fc2, inputs=fc1_spike, mem=fc2_mem, spike=fc2_spike)
            fc2_sumspike += fc2_spike
            
        out = fc2_sumspike / self.time_window
        # out = torch.softmax(fc2_sumspike / self.time_window, -1)
        return out

'''
# test code for debug
import gym
env = gym.make('CarRacing-v0')

state = env.reset()
state = state.reshape(3,96,96).astype('float32')/255
device = torch.device('cpu')

net = SCNN(cfg_cnn=0,batch_size=10,device=device)
s = torch.zeros((10,4,96,96))
s[0,0:3,:,:] = torch.tensor(state)

ab,v = net(s)
print(ab[0][0],ab[1][0],v)
'''