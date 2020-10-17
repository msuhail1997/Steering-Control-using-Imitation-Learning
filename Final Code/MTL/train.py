import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
#from torchnet.meter import AverageValueMeter
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from model import *

parser = {
    'data_dir': 'track1data',
    'nb_epoch': 50,
    'test_size': 0.1,
    'learning_rate': 0.000002,
    'samples_per_epoch': 64,
    'batch_size': 32,
    'cuda': False,
    'seed': 7
}
args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def load_data(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'modified_driving_log1010-2.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed', 'speedsequence', 'steersequence'])
    X = data_df[['center', 'left', 'right', 'speedsequence']].values
    y = data_df[['steering','throttle']].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0, shuffle=True)

    return X_train, X_valid, y_train, y_valid


X_train, X_valid, y_train, y_valid = load_data(args)
transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])
train_set = TrainCarDataset4Images(X_train, y_train, args.data_dir,transformations)
valid_set = ValidCarDataset(X_valid, y_valid, args.data_dir, transformations)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4)


def toVariable(data, use_cuda):
    input, target1, speed, target2 = data
    
    input, target1, speed, target2 = Variable(input.float()), Variable(target1.float()), Variable(speed.float()), Variable(target2.float()) #, Variable(steer.float())
    if use_cuda:
        input, target1, speed, target2 = input.cuda(), target1.cuda(), speed.cuda(), target2.cuda()#, steer.cuda()
    return input, target1, speed, target2


batch_loss=[]
count=0
x_axis=[]
ground_steering_angle=[]
pred_steering_angle=[]
def train(epoch, net, dataloader, optimizer, criterion, use_cuda):
    net.train()
    train_loss = 0
    tot_loss = 0
    global batch_loss
    global count
    global x_axis

    for batch_idx, (centers, lefts, rights, center_flips) in enumerate(dataloader):
        # (hs,cs) = net.init_hiddenState()
        optimizer.zero_grad()
        centers, lefts, rights, center_flips = toVariable(centers, use_cuda), \
                                 toVariable(lefts, use_cuda), \
                                 toVariable(rights, use_cuda), \
                                 toVariable(center_flips, use_cuda)
        datas = [lefts, rights, centers, center_flips]
        for data in datas:
            imgs, target1, speed, target2 = data
            output1, output2= net(imgs, speed)#, steer[:,-5:])

            loss1 = criterion(output1, target1)
            loss2 = criterion(output2, target2)
            loss = 0.99*loss1 + 0.01*loss2
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()
            tot_loss = train_loss/((batch_idx+1)*4)

        if batch_idx % 100 == 0:
            print('Loss : %.3f '
                % (tot_loss))
            batch_loss.append(tot_loss)
            x_axis.append(count)
        count+=1

    print("Training Loss: ", tot_loss)


validb_loss=[]
count_val=0
x_axis_val=[]

def valid(epoch, net, validloader, criterion, use_cuda):
    global best_loss
    net.eval()
    valid_loss = 0
    valid_loss = 0
    global validb_loss
    global count_val
    global x_axis_val


    for batch_idx, (inputs, target1, speed, target2) in enumerate(validloader):

        inputs, target1, speed, target2 = Variable(inputs.float()), Variable(target1.float()), Variable(speed.float()), Variable(target2.float())#, Variable(steer.float())
        if use_cuda:
            inputs, target1, speed, target2 = inputs.cuda(), target1.cuda(), speed.cuda(), target2.cuda()#, steer.cuda()
        output1, output2 = net(inputs, speed)#, steer[:,-5:])

        loss1 = criterion(output1, target1)
        loss2 = criterion(output2, target2)
        loss = 0.99*loss1 + 0.01*loss2

        valid_loss += loss.item()#data[0]

        avg_valid_loss = valid_loss/(batch_idx+1)

        if batch_idx % 100 == 0:
            print('Valid Loss : %.3f '
                % (valid_loss/(batch_idx+1)))
            blv=(valid_loss)/((batch_idx+1))
            validb_loss.append(blv)
            x_axis_val.append(count_val)
        count_val+=1

    print("Validation Loss: ", avg_valid_loss)
    if avg_valid_loss <= best_loss:
        best_loss = avg_valid_loss
        print('Best epoch: ' + str(epoch))
        state = {
            'net': net.module  if args.cuda else net,
        }
        torch.save(state, './model2000.h5')

net = MTL()
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
# print(args.cuda)
if args.cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    cudnn.benchmark = True

criterion = nn.MSELoss()

best_loss = 1.999


for epoch in range(0,3):
    print('\nEpoch: %d' % epoch)
    train(epoch, net, train_loader, optimizer, criterion, args.cuda)
    valid(epoch, net, valid_loader, criterion, args.cuda)

state = {
        'net': net.module if args.cuda else net,
        }

torch.save(state, './model1000.h5')


#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$   PLOTTING GRAPHS     $$$$$$$$$$$$$$$$$$$$$#########################
# calculating steering angle
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)

for batch_idx, (centers, lefts, rights, center_flips) in enumerate(train_loader):

    centers, lefts, rights, center_flips = toVariable(centers, args.cuda), \
                             toVariable(lefts, args.cuda), \
                             toVariable(rights, args.cuda), \
                             toVariable(center_flips, args.cuda)
    datas = [lefts, rights, centers, center_flips]
    for data in datas:
        imgs, target1, speed, target2 = data
        output1, output2= net(imgs, speed)#, steer[:,-5:])
        # imgs, targets = data
        # outputs = net(imgs).view(-1)
        ground_steering_angle.append(target1.cpu().item())
        pred_steering_angle.append(output1.cpu().item())



# # Loss graph calculation
plt.plot(x_axis,batch_loss)
plt.title("Imitation Learning MTL")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.show()
plt.title("Imitation Learning MTL")
plt.plot(x_axis_val,validb_loss)
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")
plt.show()

plt.plot(ground_steering_angle[0::100])
plt.plot(pred_steering_angle[0::100])
plt.title("Imitation Learning (MTL Baseline) Steering angle")
plt.legend(['Ground Steering angle','Predicted Steering angle'])
plt.xlabel("Iterations")
plt.ylabel("Steering angle")
plt.show()

