import torch.nn as nn
import torch

def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1) 

class PupilNet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_extra = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.act_extra = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.act2 = nn.ReLU()
        self.drop1 = nn.Dropout2d()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.act4 = nn.ReLU()
        self.drop2 = nn.Dropout2d()
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.act5 = nn.ReLU()
        self.global_pool = nn.AvgPool2d(kernel_size=(2, 4))
        self.fc1 = nn.Linear(256, 64)
        self.act6 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.act_extra(self.conv_extra(x))
        x = self.drop1(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = self.drop2(self.act4(self.conv4(x)))
        x = self.act5(self.conv5(x))
        x = flatten(self.global_pool(x))
        x = self.fc2(self.act6(self.fc1(x)))
        return x
