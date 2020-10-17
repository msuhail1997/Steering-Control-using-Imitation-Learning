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


class MTL(nn.Module):
    def __init__(self):
        super(MTL, self).__init__()
        self.hidden_size = 128
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 16 x 32
            nn.Conv2d(3, 16,(3,2), stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(16, 32, (3, 5), stride=2, bias=False),
            nn.ELU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 48, (2,4), stride=(1,2), bias=False),
            nn.ELU(),
            # nn.BatchNorm2d(48),
            nn.Conv2d(48, 64, (2,4), stride=(1,2), bias=False),
            nn.ELU()
        )

        self.linear_layers1 = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=18432, out_features=72, bias=False),
            nn.ELU())

        self.linear_layers2 = nn.Sequential(
            nn.Linear(in_features=72, out_features=36, bias=False),
            nn.ELU(),
            nn.Linear(in_features=36, out_features=24, bias=False),
            nn.ELU(),
            nn.Linear(in_features=24, out_features=16, bias=False),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=1, bias=False)
        )

        self.lstm = nn.LSTM(1, 128, bias=False)
        self.lstm2 = nn.LSTM(1, 128, bias=False)
        

        self.linear_layers3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=72, bias=False),
            nn.ELU())

        self.linear_layers4 = nn.Sequential(
            nn.Linear(in_features=144, out_features=72, bias=False),
            nn.ELU(),
            nn.Linear(in_features=72, out_features=24, bias=False),
            nn.ELU(),
            nn.Linear(in_features=24, out_features=16, bias=False),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=1, bias=False),
            nn.Sigmoid()
        )

        self.linear_layers5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=72, bias=False),
            nn.ELU()

        )

        self.linear_layers6 = nn.Sequential(
            nn.Linear(in_features=216, out_features=72, bias=False),
            nn.ELU(),
            nn.Linear(in_features=72, out_features=36, bias=False),
            nn.ELU(),
            nn.Linear(in_features=36, out_features=16, bias=False),
            nn.ELU(),
            nn.Linear(in_features=16, out_features=1, bias=False)
        )

        # self.sig = nn.Sigmoid()

        self._initialize_weights()

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, image, speed):

        image = image.view(image.size(0), 3, 75, 320)
        output1 = self.conv_layers(image)

        #print('size ' + str(output1.size()))
        output1 = output1.view(output1.size(0), -1)

        output2 = self.linear_layers1(output1)
        output3 = self.linear_layers2(output2)

        speed = speed.permute(1,0)
        speed = speed.unsqueeze(2)
        # steering = steering.permute(1,0)
        # steering = steering.unsqueeze(2)
        # print(speed.shape)

        # speed = speed.view(10,image.size(0),1)
        # print(speed.shape)
        (hs, cs) = self.init_hiddenState(speed.shape[1])
        _, (output4, __) = self.lstm(speed, (hs,cs))
        # (hs1, cs1) = self.init_hiddenState(steering.shape[1])
        # _, (output8, __) = self.lstm2(steering, (hs1, cs1))


        # print(output4.shape)
        output4 = output4.squeeze(0)
        output5 = self.linear_layers3(output4)

        
        # print(output2.shape)
        # print(output4.shape)
        output6 = torch.cat((output2, output5),1)
        output7 = self.linear_layers4(output6)
        # print(oo.shape)

        # output7 = sig(oo)

        # output8 = output8.squeeze(0)
        # output9 = self.linear_layers5(output8)
        # print(output2.shape)
        # print(output9.shape)
        # output10 = torch.cat((output2, output9, output5),1)


        # output11 = self.linear_layers6(output10)

        return output3, output7

    def init_hiddenState(self, batchsize):
        h0 = torch.randn(1, batchsize, self.hidden_size, dtype=torch.float)
        c0 = torch.randn(1, batchsize, self.hidden_size, dtype=torch.float)      

        return (h0,c0)