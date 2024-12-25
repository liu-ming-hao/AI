import torch.nn as nn
import torch.nn.functional as F #引入激活函数
#模型选型

class CVFacialExpressionModel(nn.Module):
    def __init__(self,nclass):
        super(CVFacialExpressionModel,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2) #输入3通道，输出12通道，卷积核大小3，步长2
        self.bn1 = nn.BatchNorm2d(12) #批量归一化
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*5*5,1200)
        self.fc2 = nn.Linear(1200,128)
        self.fc3 = nn.Linear(128,nclass)

    #前向传播
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1,48*5*5) #展平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
