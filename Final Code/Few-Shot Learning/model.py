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

#Model Definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
        #Conv Layers
            nn.Conv2d(3, 16,(3,2), stride= (1,3) , bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.8),
            nn.Conv2d(16, 32, (5, 3), stride=(1,2), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.7),
            nn.Conv2d(32, 64, (3,3), stride=(3,2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(0.6),
            nn.Conv2d(64, 128, (2,3), stride=(1,2), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, (1,3), stride=(3,2), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Conv2d(256, 512, (2,2), bias=False),
            nn.ELU()
        )
        self.linear_layers = nn.Sequential(
            #Fully Connected Layers
            nn.Linear(in_features=14336, out_features=2048, bias=False),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=1024, bias=False),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=1024, out_features=512, bias=False),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=128, bias=False),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 8),
            nn.ELU(),
            nn.Linear(8, 1)
        )

        self.s = nn.Softmax(dim=1)
        self._initialize_weights()

    #Weight Initialization by normalizing the input with mean and standard deviation
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.05)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward_once(self, input):
        input = input.view(-1, 3, 75, 320)
        input = input.float()
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

    def forward(self, input1, input2, input3,input4,steer_list):
        #Pass the four inputs into the first set of convolution and fully connected layers
        output_1 = self.forward_once(input1)
        output_2 = self.forward_once(input2)
        output_3 = self.forward_once(input3)
        output_4 = self.forward_once(input4)
        #Concatenating the output latent vectors
        result_1 = torch.cat((output_1, output_2),1)
        result_2 = torch.cat((output_1, output_3),1)
        result_3 = torch.cat((output_1, output_4),1)
        #Passing them through another set of fully connected Layers
        i1 = self.linear1(result_1)
        i2 = self.linear1(result_2)
        i3 = self.linear1(result_3)
        #Concatenating the outputs
        i = torch.cat((i1,i2,i3),1)
        #Softmax done to get the probability vector
        softmax = self.s(i)
        output_dot = torch.bmm(softmax.view(input1.size(0), 1, 3), steer_list.view(input1.size(0), 3, 1)).view(-1)
        #Output Steering Angle
        return output_dot





class Final(nn.Module):
  def __init__(self):
    super(Final, self).__init__()
    self.linear1 = nn.Sequential(
        nn.Linear(256, 64),
        nn.ELU(),
        nn.Linear(64, 8),
        nn.ELU(),
        nn.Linear(8, 1)
    )

    self.s = nn.Softmax(dim=1)
    self._initialize_weights()

  def _initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Linear):
              init.normal(m.weight, mean=0, std=0.2)


  def forward(self,o1, o2, o3):
    i1 = self.linear1(o1)
    i2 = self.linear1(o2)
    i3 = self.linear1(o3)
    i = torch.cat((i1,i2,i3),1)
    return self.s(i)
