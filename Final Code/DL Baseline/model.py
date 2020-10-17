import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn



"""
DL Baseline network
"""
class DLBaseline(nn.Module):
    def __init__(self):
        super(DLBaseline, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 16 x 32
            nn.Conv2d(3, 32,3, bias=False), nn.MaxPool2d(2,2),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(32, 64,3,bias=False),nn.MaxPool2d(2,2),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(64, 128,3,bias=False),nn.MaxPool2d(2,2),
            nn.ELU(),
            nn.Conv2d(128, 256,3,bias=False),nn.MaxPool2d(2,2),
            nn.ELU(),
        )

        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=9216, out_features=120, bias=False),
            nn.ELU(),
            nn.Linear(in_features=120, out_features=20, bias=False),
            nn.ELU(),
            nn.Linear(in_features=20, out_features=1, bias=False),
        )

        self._initialize_weights()

    def _initialize_weights(self):

        """
        Weight initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):

        input = input.view(input.size(0), 3, 75, 320)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output
