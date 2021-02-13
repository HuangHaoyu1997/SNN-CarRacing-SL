import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    Convolutional Neural Network for Classification
    """
    def __init__(self, input_channel,output_channel):
        super(ConvNet, self).__init__()
        self.cnn_base = nn.Sequential(                   # input shape (4, 96, 96)
            nn.Conv2d(input_channel, 8, kernel_size=4, stride=2),
            nn.ReLU(),  
            nn.Conv2d(8, 16, kernel_size=3, stride=2),   # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),# (128, 3, 3)
            nn.ReLU(),
        )                                                # output shape (256, 1, 1)
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, output_channel))
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

class ConvAgent(nn.Module):
    """
    Convolutional Neural Network Agent for PPO
    """
    def __init__(self, img_stack):
        super(ConvAgent, self).__init__()
        self.cnn_base = nn.Sequential(                   # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  
            nn.Conv2d(8, 16, kernel_size=3, stride=2),   # (8, 47, 47)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # (64, 5, 5)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),# (128, 3, 3)
            nn.ReLU(),
        )                                                # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1 # alpha和beta定义了1个beta分布，用以采样动作向量
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

class ActFun(torch.autograd.Function):
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

def mem_update(ops, inputs, spike, mem, thr=0.3, v_th=0., decay=1., activation=None):
    '''
    fv: lateral connections (option)
    decay: membrane decay functions
    SNN的更新函数，当成RNN来理解
    ops：可以是卷积操作，或MLP操作
    '''
    state = ops(inputs)  # 当前状态,
    # mem是膜电位，spike=1即上一状态发放了脉冲，则mem减去一个发放阈值thr
    # spike=0则上一时刻未发放脉冲，则膜电位继续累积
    # Here we use a firing-and-substration strategy and formulize the learnable decay parameters
    
    mem = state + mem * (1 - spike) * decay# .clamp(min=0., max=1.)
    # mem = state + (mem - spike * thr) * decay# .clamp(min=0., max=1.)
    now_spike = act_fun(mem - thr)
    return mem, now_spike.float()

class SNN(nn.Module):
    def __init__(self, hidden_units=32, input_size=784, output_size=10, wins=10, batch_size=100,v_th_scales=0.8,cfg_layer=[28*28, 128, 128, 10]):  # 中间两个值应该与hidden_unit相等
        '''
        3层SNN全连接网络,输入x只含1个样本，重复wins次输入SNN
        '''
        super(SNN, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.wins = wins
        self.batch_size = batch_size # 对于RL算法而言batch=1更灵活
        self.cfg_layer = cfg_layer
        self.fc1 = nn.Linear(self.cfg_layer[0], self.cfg_layer[1], bias=True)
        self.fc2 = nn.Linear(self.cfg_layer[1], self.cfg_layer[2], bias=True)
        self.fc3 = nn.Linear(self.cfg_layer[2], self.cfg_layer[3], bias=True)  # linear readout layers

        # Learnable threshold
        self.v_th1 = nn.Parameter(v_th_scales * torch.rand(hidden_units, device=device))  # tensor变parameter，可训练
        self.v_th2 = nn.Parameter(v_th_scales * torch.rand(hidden_units, device=device))
        self.v_th3 = nn.Parameter(v_th_scales * torch.rand(output_size, device=device))

        # Learnable decay
        self.decay1 = nn.Parameter(torch.rand(self.cfg_layer[1], device=device))  # tensor变parameter，可训练
        self.decay2 = nn.Parameter(torch.rand(self.cfg_layer[2], device=device))
        self.decay3 = nn.Parameter(torch.rand(self.cfg_layer[3], device=device))

    def forward(self, x):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.cfg_layer[1], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.cfg_layer[2], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.cfg_layer[3], device=device)
        
        for step in range(self.wins):
            # x = x > torch.rand(x.size(), device = device) # 概率编码
            h1_mem, h1_spike = mem_update(fc=self.fc1, inputs=x.to(device), spike=h1_spike, mem=h1_mem, thr=0.5, v_th=self.v_th1, decay=self.decay1)

            h2_mem, h2_spike = mem_update(fc=self.fc2, inputs=h1_spike, spike=h2_spike, mem=h2_mem, thr=0.5, v_th=self.v_th2, decay=self.decay2)

            h2_sumspike = h2_sumspike + h2_spike # 128维

        outs = self.fc3(h2_sumspike/self.wins)  # readout layers
        # print('out,',outs)
        outs = F.softmax(outs, dim=1)
        return outs

class SNN_(nn.Module):
    def __init__(self, hidden_units=32, input_size=784, output_size=10, wins=10, batch_size=100,v_th_scales=0.8,cfg_layer=[28*28, 128, 128, 10]):  # 中间两个值应该与hidden_unit相等
        '''
        3层SNN全连接网络,输入x内部具有wins个样本，按照顺序依次输入SNN
        '''
        super(SNN_, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.wins = wins
        self.batch_size = batch_size # 对于RL算法而言batch=1更灵活
        self.cfg_layer = cfg_layer
        self.fc1 = nn.Linear(self.cfg_layer[0], self.cfg_layer[1], bias=True)
        self.fc2 = nn.Linear(self.cfg_layer[1], self.cfg_layer[2], bias=True)
        self.fc3 = nn.Linear(self.cfg_layer[2], self.cfg_layer[3], bias=True)  # linear readout layers

        # Learnable threshold
        self.v_th1 = nn.Parameter(v_th_scales * torch.rand(hidden_units, device=device))  # tensor变parameter，可训练
        self.v_th2 = nn.Parameter(v_th_scales * torch.rand(hidden_units, device=device))
        self.v_th3 = nn.Parameter(v_th_scales * torch.rand(output_size, device=device))

        # Learnable decay
        self.decay1 = nn.Parameter(torch.rand(self.cfg_layer[1], device=device))  # tensor变parameter，可训练
        self.decay2 = nn.Parameter(torch.rand(self.cfg_layer[2], device=device))
        self.decay3 = nn.Parameter(torch.rand(self.cfg_layer[3], device=device))

    def forward(self, x):
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.cfg_layer[1], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.cfg_layer[2], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.cfg_layer[3], device=device)
        
        for step in range(self.wins):
            x_ = x[:,step,:,:].view(-1,28*28).to(device)
            # x = x > torch.rand(x.size(), device = device) # 概率编码
            h1_mem, h1_spike = mem_update(fc=self.fc1, inputs=x_, spike=h1_spike, mem=h1_mem, thr=0.5, v_th=self.v_th1, decay=self.decay1)

            h2_mem, h2_spike = mem_update(fc=self.fc2, inputs=h1_spike, spike=h2_spike, mem=h2_mem, thr=0.5, v_th=self.v_th2, decay=self.decay2)

            h2_sumspike = h2_sumspike + h2_spike # 128维

        outs = self.fc3(h2_sumspike/self.wins)  # readout layers
        outs = F.softmax(outs, dim=1)
        return outs

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    def __init__(self,batch_size,device):
        super(SCNN, self).__init__()

        self.batch_size = batch_size
        self.device = device

        self.conv1 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        
        # value net head
        self.v1 = nn.Linear(256, 100)
        self.v2 = nn.Linear(100, 1)
        # action net head
        self.fc = nn.Linear(256,100)
        self.alpha_head = nn.Linear(100,3)
        self.beta_head = nn.Linear(100,3)

        self.thr_a = nn.Parameter(torch.rand(3, device=self.device))  # tensor变parameter，可训练
        self.thr_b = nn.Parameter(torch.rand(3, device=self.device))
        self.thr_v = nn.Parameter(torch.rand(1, device=self.device))

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    def forward(self, input, time_window = 16):
        c1_mem = c1_spike = torch.zeros(self.batch_size, 8, 47, 47, device=self.device) # 第1卷积层output feature map size=[batch,channel,width,height]
        c2_mem = c2_spike = torch.zeros(self.batch_size, 16, 23, 23, device=self.device) # 第2卷积层output feature map size=[batch,channel,width,height]
        c3_mem = c3_spike = torch.zeros(self.batch_size, 32, 11, 11, device=self.device) # 第3卷积层output feature map size=[batch,channel,width,height]
        c4_mem = c4_spike = torch.zeros(self.batch_size, 64, 5, 5, device=self.device) # 第1卷积层output feature map size=[batch,channel,width,height]
        c5_mem = c5_spike = torch.zeros(self.batch_size, 128, 3, 3, device=self.device) # 第2卷积层output feature map size=[batch,channel,width,height]
        c6_mem = c6_spike = torch.zeros(self.batch_size, 256, 1, 1, device=self.device) # 第3卷积层output feature map size=[batch,channel,width,height]
        
        v1_mem = v1_spike = v1_sumspike = torch.zeros(self.batch_size, 100, device=self.device) 
        v2_mem = v2_spike = v2_sumspike = torch.zeros(self.batch_size, 1, device=self.device) 
        
        fc_mem = fc_spike = fc_sumspike = torch.zeros(self.batch_size, 100, device=self.device) 
        a_mem = a_spike = a_sumspike = torch.zeros(self.batch_size, 3, device=self.device) 
        b_mem = b_spike = b_sumspike = torch.zeros(self.batch_size, 3, device=self.device) 

        

        for step in range(time_window): # 仿真时间，即发放次数
            x = input > torch.rand(input.size(), device=self.device) # prob. firing

            c1_mem, c1_spike = mem_update(ops=self.conv1, inputs=x.float(), mem=c1_mem, spike=c1_spike)
            c2_mem, c2_spike = mem_update(ops=self.conv2, inputs=c1_spike, mem=c2_mem, spike=c2_spike)
            c3_mem, c3_spike = mem_update(ops=self.conv3, inputs=c2_spike, mem=c3_mem, spike=c3_spike)
            c4_mem, c4_spike = mem_update(ops=self.conv4, inputs=c3_spike, mem=c4_mem, spike=c4_spike)
            c5_mem, c5_spike = mem_update(ops=self.conv5, inputs=c4_spike, mem=c5_mem, spike=c5_spike)
            c6_mem, c6_spike = mem_update(ops=self.conv6, inputs=c5_spike, mem=c6_mem, spike=c6_spike)
            
            c6_flatten = c6_spike.view(self.batch_size, -1)
            # value net inference
            v1_mem, v1_spike = mem_update(ops=self.v1, inputs=c6_flatten, mem=v1_mem, spike=v1_spike)
            v1_sumspike += v1_spike
            v2_mem, v2_spike = mem_update(ops=self.v2, inputs=v1_spike, thr=self.thr_v ,mem=v2_mem, spike=v2_spike)
            v2_sumspike += v2_spike

            # action net inference
            fc_mem, fc_spike = mem_update(ops=self.fc, inputs=c6_flatten, mem=fc_mem, spike=fc_spike)
            fc_sumspike += fc_spike
            a_mem, a_spike = mem_update(ops=self.alpha_head, inputs=fc_spike, thr=self.thr_a, mem=a_mem, spike=a_spike)
            a_sumspike += a_spike
            b_mem, b_spike = mem_update(ops=self.beta_head, inputs=fc_spike, thr=self.thr_b, mem=b_mem, spike=b_spike)
            b_sumspike += b_spike
        
        v = v2_sumspike / time_window
        alpha = torch.sigmoid(a_mem)+1 # a_sumspike / time_window + 1
        # print(torch.sigmoid(b_mem)[0])
        beta = torch.sigmoid(b_mem)+1 # b_sumspike / time_window + 1

        return (alpha,beta), v

class SCNN_discrete(nn.Module):
    def __init__(self,batch_size,device):
        super(SCNN_discrete, self).__init__()
        self.batch_size = batch_size
        self.device = device

        self.conv1 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        
        self.fc = nn.Linear(256,100)
        
        self.value_head = nn.Linear(100, 1) # value net head
        self.action_head = nn.Linear(100, 3) # action net head

        self.v = nn.Linear(1, 1)

        self.thr_a = nn.Parameter(torch.rand(3, device=self.device))  # tensor变parameter，可训练
        self.thr_v = nn.Parameter(torch.rand(1, device=self.device))
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    def forward(self, input, time_window = 16):
        c1_mem = c1_spike = torch.zeros(self.batch_size, 8, 47, 47, device=self.device) # 第1卷积层output feature map size=[batch,channel,width,height]
        c2_mem = c2_spike = torch.zeros(self.batch_size, 16, 23, 23, device=self.device)
        c3_mem = c3_spike = torch.zeros(self.batch_size, 32, 11, 11, device=self.device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, 64, 5, 5, device=self.device)
        c5_mem = c5_spike = torch.zeros(self.batch_size, 128, 3, 3, device=self.device)
        c6_mem = c6_spike = torch.zeros(self.batch_size, 256, 1, 1, device=self.device)
        
        fc_mem = fc_spike = fc_sumspike = torch.zeros(self.batch_size, 100, device=self.device) 
        v_mem = v_spike = v_sumspike = torch.zeros(self.batch_size, 1, device=self.device)
        a_mem = a_spike = a_sumspike = torch.zeros(self.batch_size, 3, device=self.device)

        for step in range(time_window): # 仿真时间，即发放次数
            x = input > torch.rand(input.size(), device=self.device) # prob. firing

            c1_mem, c1_spike = mem_update(ops=self.conv1, inputs=x.float(), mem=c1_mem, spike=c1_spike)
            c2_mem, c2_spike = mem_update(ops=self.conv2, inputs=c1_spike, mem=c2_mem, spike=c2_spike)
            c3_mem, c3_spike = mem_update(ops=self.conv3, inputs=c2_spike, mem=c3_mem, spike=c3_spike)
            c4_mem, c4_spike = mem_update(ops=self.conv4, inputs=c3_spike, mem=c4_mem, spike=c4_spike)
            c5_mem, c5_spike = mem_update(ops=self.conv5, inputs=c4_spike, mem=c5_mem, spike=c5_spike)
            c6_mem, c6_spike = mem_update(ops=self.conv6, inputs=c5_spike, mem=c6_mem, spike=c6_spike)
            
            c6_flatten = c6_spike.view(self.batch_size, -1)
            fc_mem, fc_spike = mem_update(ops=self.fc, inputs=c6_flatten, mem=fc_mem, spike=fc_spike)
            fc_sumspike += fc_spike

            v_mem, v_spike = mem_update(ops=self.value_head, inputs=fc_spike, thr=self.thr_v ,mem=v_mem, spike=v_spike)
            v_sumspike += v_spike
            a_mem, a_spike = mem_update(ops=self.action_head, inputs=fc_spike, thr=self.thr_a, mem=a_mem, spike=a_spike)
            a_sumspike += a_spike
        
        value = self.v(v_sumspike / time_window)
        action = torch.softmax(a_sumspike / time_window, -1)

        return action, value

class SCNN_classification(nn.Module):
    def __init__(self,input_channel,output_channel,batch_size,device):
        super(SCNN_classification, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.output_channel = output_channel
        self.conv1 = nn.Conv2d(input_channel, 8, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        
        self.fc1 = nn.Linear(256,100)
        self.fc2 = nn.Linear(100, output_channel)

        # self.thr = nn.Parameter(torch.rand(output_channel, device=self.device))  # tensor变parameter，可训练
    
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
    
    def forward(self, input, time_window = 6):
        c1_mem = c1_spike = torch.zeros(self.batch_size, 8, 47, 47, device=self.device) # 第1卷积层output feature map size=[batch,channel,width,height]
        c2_mem = c2_spike = torch.zeros(self.batch_size, 16, 23, 23, device=self.device)
        c3_mem = c3_spike = torch.zeros(self.batch_size, 32, 11, 11, device=self.device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, 64, 5, 5, device=self.device)
        c5_mem = c5_spike = torch.zeros(self.batch_size, 128, 3, 3, device=self.device)
        c6_mem = c6_spike = torch.zeros(self.batch_size, 256, 1, 1, device=self.device)
        
        fc1_mem = fc1_spike = fc1_sumspike = torch.zeros(self.batch_size, 100, device=self.device) 
        fc2_mem = fc2_spike = fc2_sumspike = torch.zeros(self.batch_size, self.output_channel, device=self.device)

        for step in range(time_window): # 仿真时间，即发放次数
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

            fc2_mem, fc2_spike = mem_update(ops=self.fc2, inputs=fc1_spike, thr=0.3 ,mem=fc2_mem, spike=fc2_spike)
            fc2_sumspike += fc2_spike
            
        out = torch.softmax(fc2_sumspike / time_window, -1)

        return out

'''
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