<<<<<<< HEAD
import torch
import torch.nn as nn

class CNN(nn.Module):
    # 定义模型结构
    def __init__(self):
        '''
        一般把网络中可学习参数的层（如全连接，卷积层等）放入构造函数init中，当然也可以把不具有参数的层放在里面
        不具有可学习参数的层（ReLU，dropout，BatchNormalization）如果不放在构造函数里，则在forward方法中可以使用nn.functional来代替
        '''
        
        super(CNN,self).__init__()  # 调用父类的构造函数
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            # 卷积层
            nn.Conv2d(
                in_channels = 3,    # 输入图片的通道数
                out_channels = 16,  # 输出图片的通道数
                kernel_size = 5,    # 过滤器的大小
                stride = 1,         # 步长
                padding = 2,        # 单边填充宽度 
            ),
            # 输出大小 = (input_size+2*padding-kernal_size)/stride + 1
            
            # 激活函数。池化层已经引入了非线性性质，为什么还要加入激活函数
            nn.ReLU(),
            
            # 池化层
            nn.MaxPool2d(kernel_size=2)
            # 输出大小 = input_size / kernal_size
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3,1,1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2)
                                   )
        
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,3,1,1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2)
                                   )
        
        # 全连接输出层
        self.output = nn.Linear(in_features=128*8*8 , out_features=2)
        
    # 定义前向传播，也就是层之间的连接关系
    # forward方法是必须重写的
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)   # 对张量x进行reshape，使其适合后续全连接层
        '''
        x.size(0)：这部分获取了x的第一个维度的大小，通常代表一个batch中的样本数量
        -1：表示自动计算该维度的带线啊哦
        view()：函数用于改变张量的形状，而不改变其数据
        
        假设 x 的初始形状为 (batch_size, channels, height, width)，经
        过卷积层处理后，可能会变成类似于 (batch_size, num_filters, new_height, new_width) 的形状。
        使用 x.view(x.size(0), -1) 后，x 的形状将变为 (batch_size, num_filters * new_height * new_width)，
        即将所有的空间维度展平为一个一维向量。
        '''
        output = self.output(x)
=======
import torch
import torch.nn as nn

class CNN(nn.Module):
    # 定义模型结构
    def __init__(self):
        '''
        一般把网络中可学习参数的层（如全连接，卷积层等）放入构造函数init中，当然也可以把不具有参数的层放在里面
        不具有可学习参数的层（ReLU，dropout，BatchNormalization）如果不放在构造函数里，则在forward方法中可以使用nn.functional来代替
        '''
        
        super(CNN,self).__init__()  # 调用父类的构造函数
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            # 卷积层
            nn.Conv2d(
                in_channels = 3,    # 输入图片的通道数
                out_channels = 16,  # 输出图片的通道数
                kernel_size = 5,    # 过滤器的大小
                stride = 1,         # 步长
                padding = 2,        # 单边填充宽度 
            ),
            # 输出大小 = (input_size+2*padding-kernal_size)/stride + 1
            
            # 激活函数。池化层已经引入了非线性性质，为什么还要加入激活函数
            nn.ReLU(),
            
            # 池化层
            nn.MaxPool2d(kernel_size=2)
            # 输出大小 = input_size / kernal_size
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3,1,1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2)
                                   )
        
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,3,1,1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2)
                                   )
        
        # 全连接输出层
        self.output = nn.Linear(in_features=128*8*8 , out_features=2)
        
    # 定义前向传播，也就是层之间的连接关系
    # forward方法是必须重写的
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)   # 对张量x进行reshape，使其适合后续全连接层
        '''
        x.size(0)：这部分获取了x的第一个维度的大小，通常代表一个batch中的样本数量
        -1：表示自动计算该维度的带线啊哦
        view()：函数用于改变张量的形状，而不改变其数据
        
        假设 x 的初始形状为 (batch_size, channels, height, width)，经
        过卷积层处理后，可能会变成类似于 (batch_size, num_filters, new_height, new_width) 的形状。
        使用 x.view(x.size(0), -1) 后，x 的形状将变为 (batch_size, num_filters * new_height * new_width)，
        即将所有的空间维度展平为一个一维向量。
        '''
        output = self.output(x)
>>>>>>> a4e6ace (finish CNN)
        return output 