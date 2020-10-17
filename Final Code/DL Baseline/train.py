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
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model import *

parser = {
    'data_dir': '/home/suhail/Desktop/track2data',
    'nb_epoch': 7,
    'test_size': 0.1,
    'learning_rate': 1e-4,
    'samples_per_epoch': 64,
    'batch_size': 32,
    'cuda': True,
    'seed': 7
}
args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def load_data(args):
    """
    Data loading
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data_df = data_df[data_df['left'] != 'left']
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0, shuffle=True)

    return X_train, X_valid, y_train, y_valid


"""
Data splitting/loading
"""
X_train, X_valid, y_train, y_valid = load_data(args)
transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])
train_set = TrainCarDataset4Images(X_train, y_train, args.data_dir,transformations)
valid_set = ValidCarDataset(X_valid, y_valid, args.data_dir,transformations)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4)



def toCUDA(data, use_cuda):

    """
    wrapping the tensors with a Variable function
    """
    input, target = data
    input, target = Variable(input.float()), Variable(target.float())
    if use_cuda:
        input, target = input.cuda(), target.cuda()

    return input, target
batch_loss=[]
count=0
x_axis=[]
ground_steering_angle=[]
pred_steering_angle=[]
x=0


def train(epoch, net, dataloader, optimizer, criterion, use_cuda):

    """
    Training loop
    """
    net.train()
    train_loss = 0
    global batch_loss
    global count
    global x_axis
    global x
    for batch_idx, (centers, lefts, rights, center_flips) in enumerate(dataloader):
        count=0
        optimizer.zero_grad()
        centers, lefts, rights, center_flips = toCUDA(centers, use_cuda), \
                                 toCUDA(lefts, use_cuda), \
                                 toCUDA(rights, use_cuda), \
                                 toCUDA(center_flips, use_cuda)
        datas = [lefts, rights, centers, center_flips]
        for data in datas:
            imgs, targets = data
            outputs = net(imgs).view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            count+=1
            """
            Training with only 4000 images
            """
            if(count>4000):
                break
        if batch_idx % 100 == 0:
            print('Loss : %.3f '
                % (train_loss/((batch_idx+1)*4)))
            bl=(train_loss)/((batch_idx+1)*4)
            batch_loss.append(bl)
            x_axis.append(x)
        x+=1


validb_loss=[]
count_val=0
x_axis_val=[]


def valid(epoch, net, validloader, criterion, use_cuda):

    """
    validation loop
    """
    global best_loss
    net.eval()
    valid_loss = 0
    valid_loss = 0
    global validb_loss
    global count_val
    global x_axis_val
    for batch_idx, (inputs, targets) in enumerate(validloader):
        inputs, targets = Variable(inputs.float()), Variable(targets.float())
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs).view(-1)
        loss = criterion(outputs, targets)
        valid_loss += loss.item()#data[0]
        avg_valid_loss = valid_loss/(batch_idx+1)
        if batch_idx % 100 == 0:
            print('Valid Loss : %.3f '
                % (valid_loss/(batch_idx+1)))
            blv=(valid_loss)/((batch_idx+1))
            validb_loss.append(blv)
            x_axis_val.append(count_val)
        count_val+=1



net = DLBaseline()


if args.cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,weight_decay=1e-5)


for epoch in range(0,4):
    print('\nEpoch: %d' % epoch)
    train(epoch, net, train_loader, optimizer, criterion, args.cuda)
    valid(epoch, net, valid_loader, criterion, args.cuda)



"""
Plotting the train and steering angle graphs
"""
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
for batch_idx, (centers, lefts, rights, center_flips) in enumerate(train_loader):
    centers, lefts, rights, center_flips = toCUDA(centers, args.cuda), \
                             toCUDA(lefts, args.cuda), \
                             toCUDA(rights, args.cuda), \
                             toCUDA(center_flips, args.cuda)
    datas = [lefts, rights, centers, center_flips]
    for data in datas:
        imgs, targets = data
        outputs = net(imgs).view(-1)
        ground_steering_angle.append(targets.cpu().item())
        pred_steering_angle.append(outputs.cpu().item())

plt.plot(x_axis,batch_loss)
plt.title("Imitation Learning (DL Baseline)")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.show()
plt.title("Imitation Learning (DL Baseline)")
plt.plot(x_axis_val,validb_loss)
plt.xlabel("Iterations")
plt.ylabel("Validation Loss")
plt.show()
plt.plot(ground_steering_angle[0::100])
plt.plot(pred_steering_angle[0::100])
plt.title("Imitation Learning (DL Baseline) Steering angle")
plt.legend(['Ground Steering angle','Predicted Steering angle'])
plt.xlabel("Iterations")
plt.ylabel("Steering angle")
plt.show()

state = {
        'net': net.module if args.cuda else net,
        }

torch.save(state, './modeltrack2-random-4000.h5')
