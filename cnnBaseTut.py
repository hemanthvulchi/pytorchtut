import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.conv2d(6,16,3)
        self.fc1 = nn.linear(16*6*6,120)
        self.fc2 - nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    
    def forward(self,x):
        x=F.max_pool2