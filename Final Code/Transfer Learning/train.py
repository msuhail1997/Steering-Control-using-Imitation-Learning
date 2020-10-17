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




import torchvision.models as models
# ResnNet34 Pretrained imagenet
resnet34=models.resnet34(pretrained=True)
num_ftrs = resnet34.fc.in_features
resnet34.fc = nn.Linear(512,1)


parser = {
    'data_dir': '/home/moji/Desktop/selfdriving-car-simulator/track2data/track2data',
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



if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)


def load(args):
    """
    Loads training data and split it into training and validationation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data_df = data_df[data_df['left'] != 'left']
    #data_df.drop(data_df.index[0])
    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['center', 'left', 'right']].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    #now we can split the data into a training (80), testing(20), and validationation set
    #thanks scikit learn
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=args.test_size, random_state=0, shuffle=True)

    return X_train, X_validation, y_train, y_validation



X_train, X_validation, y_train, y_validation = load(args)
transformations = transforms.Compose([ transforms.ToPILImage()])
#Dataset Creation
train_set = TrainCarDataset4Images(X_train, y_train, args.data_dir,transformations)
validation_set = ValidCarDataset(X_validation, y_validation, args.data_dir, transformations)
#The Dataloader Creation
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True, num_workers=4)


def toCUDA(data, use_cuda):
    """
    MAkes the input variable a CUDA Variable
    """
    input, target = data
    input, target = Variable(input.float()), Variable(target.float())
    if use_cuda:
        input, target = input.cuda(), target.cuda()

    return input, target
batch_loss=[]
x_axis=[]
ground_steering_angle=[]
pred_steering_angle=[]


count_tr =0

def train(epoch, net, dataloader, optimizer, criterion, use_cuda):
    """
    Function which trains the network each epoch
    """
    net.train()
    train_loss = 0
    global batch_loss
    global x_axis
    global count_tr
    for batch_idx, (centers, lefts, rights, center_flips) in enumerate(dataloader):

        optimizer.zero_grad()
        centers, lefts, rights, center_flips = toCUDA(centers, use_cuda), \
                                 toCUDA(lefts, use_cuda), \
                                 toCUDA(rights, use_cuda), \
                                 toCUDA(center_flips, use_cuda)
        datas = [lefts, rights, centers, center_flips]

        for data in datas:
            imgs, targets = data
            # print(imgs.size())
            imgs = imgs.view(imgs.size(0),3,224,224)
            outputs = net(imgs).view(-1)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            count_tr+=1

            train_loss += loss.item()
        # if(count>4000):
        #     break

        if batch_idx % 100 == 0:
            print('Loss : %.3f '
                % (train_loss/((batch_idx+1)*3)))
            bl=(train_loss)/((batch_idx+1)*3)
            batch_loss.append(bl)
            x_axis.append(count_tr)




validationb_loss=[]
count_val=0
x_axis_val=[]
def validation(epoch, net, validationloader, criterion, use_cuda):
    """
    Function to run validationation
    """
    global best_loss
    net.eval()
    validation_loss = 0
    validation_loss = 0
    global validationb_loss
    global count_val
    global x_axis_val
    for batch_idx, (inputs, targets) in enumerate(validationloader):

        inputs, targets = Variable(inputs.float()), Variable(targets.float())


        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs).view(-1)

        loss = criterion(outputs, targets)

        validation_loss += loss.cpu().item()#data[0]

        avg_validation_loss = validation_loss/(batch_idx+1)

        if batch_idx % 10 == 0:
            # print('validation Loss : %.3f '
                # % (validation_loss/(batch_idx+1)))
            blv=(validation_loss)/((batch_idx+1)*3)
            validationb_loss.append(blv)
            x_axis_val.append(count_val)
        count_val+=1


net=resnet34
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,weight_decay=1e-5)

if args.cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    cudnn.benchmark = True

criterion = nn.MSELoss()

best_loss = 1


#Training and validationation Loop
for epoch in range(0,15):
    #optimizer = lr_scheduler(optimizer, epoch, lr_decay_epoch=args.lr_decay_epoch)
    print('\nEpoch: %d' % epoch)
    train(epoch, net, train_loader, optimizer, criterion, args.cuda)
    validation(epoch, net, validation_loader, criterion, args.cuda)



#steering angle predictions
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



#Loss graph calculation
#Training curve geberation
plt.plot(x_axis,batch_loss)
plt.title("Behavioral Cloning (Transfer Learning)")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.show()
#Validation curve generation
plt.title("Behavioral cloning (Transfer Learning)")
plt.plot(x_axis_val,validationb_loss)
plt.xlabel("Iterations")
plt.ylabel("validationation Loss")
plt.show()
#Ground VsPredicted Policy generation
plt.plot(ground_steering_angle[0::100])
plt.plot(pred_steering_angle[0::100])
plt.title("Transfer Learning Steering angle")
plt.legend(['Ground Steering angle','Predicted Steering angle'])
plt.xlabel("Iterations")
plt.ylabel("Steering angle")
plt.show()



#Saves model
state = {
        'net': net.module if args.cuda else net,
        }


torch.save(state, './modeltrack2transfer.h5')
