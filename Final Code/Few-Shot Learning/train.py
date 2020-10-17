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
from PIL import Image
import PIL
from model import *
import matplotlib.image as mpimg
import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
from torch.autograd import Variable
from model import *
import torchvision.transforms as transforms
import utils
import matplotlib.image as mpimg


#Server Initialization
sio = socketio.Server()
app = Flask(__name__)

#Minimum and Maximum speed for the vehicle
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED
parser = {
    'data_dir': '/home/raghav/Documents/track2data',
    'data_dir2': '/home/raghav/Documents/Spring2020/DLDS/Relation-Network/track1data',
    'nb_epoch': 7,
    'test_size': 0.1,
    'learning_rate': 1e-5   ,
    'samples_per_epoch': 64,
    'batch_size': 32,
    'cuda': True,
    'seed': 7
}
global BATCH_SIZE
BATCH_SIZE = 16
args = argparse.Namespace(**parser)
args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
batch_loss=[]
count=0
x_axis=[]
ground_steering_angle=[]
pred_steering_angle=[]
def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable for the train terrain
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data_df = data_df[data_df['left'] != 'left']
    #Reads the CSV file and generates the dataframe for the test terrain
    data_df1 = pd.read_csv(os.path.join(os.getcwd(), args.data_dir2, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X1 = data_df1[['center', 'left', 'right']].values
    y1 = data_df1['steering'].values
    #Splitting the data into Training(80%), Testing(20%)
    _, X_valid, __, y_valid = train_test_split(X1, y1, test_size=args.test_size, shuffle=True)
    return X, X_valid, y, y_valid


#Dataloader Initialization
X_train, X_valid, y_train, y_valid = load_data(args)
transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])
#Create the training and the validation set
train_set = TrainCarDataset4Images(X_train, y_train, args.data_dir,transformations)
valid_set = ValidCarDataset(X_valid, y_valid, args.data_dir2, transformations)
#Creating the dataloader for the training and validation set
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

def toCUDA(data, use_cuda):
    input, target = data
    input, target = Variable(input.float()), Variable(target.float())
    if use_cuda:
        input, target = input.cuda(), target.cuda()

    return input, target


class SiameseNetworkDataset():
    def __init__(self,transform=None,should_invert=True):
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self):
        #These are the  fixed images from the test terrain that is used for Few-Shot Learning
        #Image 0
        img0path = '/home/raghav/Documents/Spring2020/DLDS/time/center_2019_04_02_18_05_57_110.jpg'
        img0 =  mpimg.imread(img0path)
        global steering_img0
        #Corresponding steering angle of the image
        steering_img0 = 0
        #Image 1
        img1path = '/home/raghav/Documents/Spring2020/DLDS/time/center_2019_04_02_19_25_53_860.jpg'
        img1 =  mpimg.imread(img1path)
        global steering_img1
        steering_img1 = -0.6000001
        #Image 2
        img2path = '/home/raghav/Documents/Spring2020/DLDS/time/center_2019_04_02_19_28_27_725.jpg'
        global steering_img2
        steering_img2 = 0.6500001
        img2 =  mpimg.imread(img2path)

        #Performing preprocessing on the images
        img0 = preprocess(img0)
        img1 = preprocess(img1)
        img2 = preprocess(img2)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

#Support Set that is used for training containing a few samples of the test terrain
class SupportSet():

    def __init__(self,transform=None,should_invert=True):
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self):
        support1 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_26_17_451.jpg'
        img0 =  mpimg.imread(support1)
        global support_steering_image0
        support_steering_image0 = -0.35

        support2 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_28_31_813.jpg'
        img1 =  mpimg.imread(support2)
        global support_steering_image1
        support_steering_image1 = 0.5

        support3 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_28_34_297.jpg'
        img2 =  mpimg.imread(support3)
        global support_steering_image2
        support_steering_image2 = 0.2

        support4 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_28_48_786.jpg'
        img3 =  mpimg.imread(support4)
        global support_steering_image3
        support_steering_image3 = -0.75

        support5 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_29_23_193.jpg'
        img4 =  mpimg.imread(support5)
        global support_steering_image4
        support_steering_image4 = 0

        support6 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_30_12_391.jpg'
        img5 =  mpimg.imread(support6)
        global support_steering_image5
        support_steering_image5 = -0.05

        support7 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_30_59_269.jpg'
        img6 =  mpimg.imread(support7)
        global support_steering_image6
        support_steering_image6 = 0.05

        support8 = '/home/raghav/Documents/Spring2020/DLDS/time/images/center_2019_04_02_19_26_20_131.jpg'
        img7 =  mpimg.imread(support8)
        global support_steering_image7
        support_steering_image7 = 0

        img0 = preprocess(img0)
        img1 = preprocess(img1)
        img2 = preprocess(img2)
        img3 = preprocess(img3)
        img4 = preprocess(img4)
        img5 = preprocess(img5)
        img6 = preprocess(img6)
        img7 = preprocess(img7)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)
            img6 = self.transform(img6)
            img7 = self.transform(img7)
        #Returns the images and the corresponding steering angles
        return ([img0, img1, img2, img3, img4, img5, img6, img7] ,[support_steering_image0,support_steering_image1,support_steering_image2,support_steering_image3,support_steering_image4,support_steering_image5,support_steering_image6,support_steering_image7])


class Similarity(nn.Module):
  def __init__(self):
    super(Similarity, self).__init__()
    self.linear = nn.Sequential(
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 8)
    )
  def forward(self, i1, i2, i3):
    o1 = self.linear(i1)
    o2 = self.Linear(i2)
    o3 = self.Linear(i3)
    return o1,o2,o3

#Initialization of the Classes and declaration of the variables
siamese_compare = SiameseNetworkDataset(transforms.ToTensor())
support_set = SupportSet(transforms.ToTensor())
SiameseNet = SiameseNetwork()
FinalNet = Final()
Simil = Similarity()
ground_steering_angle=[]
pred_steering_angle=[]
global counter
counter = 0

def train(epoch, net, dataloader, optimizer, criterion, use_cuda):
    SiameseNet.train()
    train_loss = 0
    result_expand_1 = torch.zeros(BATCH_SIZE + 1,128)
    result_expand_2 = torch.zeros(BATCH_SIZE + 1,128)
    result_expand_3 = torch.zeros(BATCH_SIZE + 1,128)
    global batch_loss
    global x_axis
    for batch_idx, (centers, lefts, rights, center_flips) in enumerate(dataloader):
        optimizer.zero_grad()
        centers, lefts, rights, center_flips = toCUDA(centers, use_cuda), \
                                 toCUDA(lefts, use_cuda), \
                                 toCUDA(rights, use_cuda), \
                                 toCUDA(center_flips, use_cuda)
        #These are the input from the cameras
        datas = [lefts, rights, centers, center_flips]
        for data in datas:
            global count
            count += 1
            #To train with fewer training samples
            if count >= 4000:
                break
            #Input images and the ground truth
            imgs, targets = data
            #The fixed images used for the few-shot learning
            img0,img1,img2 = siamese_compare.__getitem__()
            #Steering List
            image_list = [steering_img0, steering_img1, steering_img2]
            #Converting to a Tensor
            image_list = torch.FloatTensor(image_list).cuda()
            #Reshaping it to a size of 32
            img0_list = img0.repeat(imgs.size(0),1,1,1)
            img1_list = img1.repeat(imgs.size(0),1,1,1)
            img2_list = img2.repeat(imgs.size(0),1,1,1)
            image_list = image_list.repeat(imgs.size(0),1)
            #Passing the images to the network
            output_steering_angle  = SiameseNet(imgs,img0_list,img1_list,img2_list,image_list)
            loss = criterion(output_steering_angle, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Loss : %.3f '
                % (train_loss/((batch_idx+1)*3)))
            bl=(train_loss)/((batch_idx+1)*3)
            batch_loss.append(bl)
            x_axis.append(counter)
            #Support Set
            #Image  list and the steering list of the Support Set
            image_support_list, steering_support_list = support_set.__getitem__()
            image_support_list = np.stack(image_support_list)
            #Converting to a Tensor
            image_support_list = torch.FloatTensor(image_support_list).cuda()
            steering_support_list = torch.FloatTensor(steering_support_list).cuda()
            #Reshaping it to a size of 32 taking the batch size into consider
            img0_list = img0.repeat(image_support_list.size(0),1,1,1)
            img1_list = img1.repeat(image_support_list.size(0),1,1,1)
            img2_list = img2.repeat(image_support_list.size(0),1,1,1)
            image_list = [steering_img0, steering_img1, steering_img2]
            image_list = torch.FloatTensor(image_list).cuda()
            image_list = image_list.repeat(image_support_list.size(0),1)
            #Passing the images into the network
            output_steering_angle  = SiameseNet(image_support_list,img0_list,img1_list,img2_list,image_list)
            loss = criterion(output_steering_angle, steering_support_list)
            loss.backward()
            optimizer.step()


#Validation Process
validb_loss=[]
count_val=0
x_axis_val=[]
def valid(epoch, net, validloader, criterion, use_cuda):
    global best_loss
    SiameseNet.eval()
    valid_loss = 0
    valid_loss = 0
    valid_loss = 0
    valid_loss = 0
    global validb_loss
    global count_val
    global x_axis_val
    for batch_idx, (inputs, targets) in enumerate(validloader):

        inputs, targets = Variable(inputs.float()), Variable(targets.float())
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #Fixed images for few-shot learning
        img0,img1,img2 = siamese_compare.__getitem__()
        image_list = [steering_img0, steering_img1, steering_img2]
        #Converting it to a Tensor
        image_list = torch.FloatTensor(image_list).cuda()
        #Reshaping it to a size of 32 taking the batch size into consider
        img0_list = img0.repeat(inputs.size(0),1,1,1)
        img1_list = img1.repeat(inputs.size(0),1,1,1)
        img2_list = img2.repeat(inputs.size(0),1,1,1)
        image_list = image_list.repeat(inputs.size(0),1)
        #Passing the images as input
        output_steering_angle  = SiameseNet(inputs,img0_list,img1_list,img2_list,image_list)
        loss = criterion(output_steering_angle, targets)
        valid_loss += loss.item()
        avg_valid_loss = valid_loss/(batch_idx+1)
        if batch_idx % 100 == 0:
            print('Valid Loss : %.3f '
                % (valid_loss/(batch_idx+1)))
            blv=(valid_loss)/((batch_idx+1)*3)
            validb_loss.append(blv)
            x_axis_val.append(count_val)

    print("Validation Loss",avg_valid_loss)
    #Saving the model
    if avg_valid_loss <= best_loss:
        best_loss = avg_valid_loss
        print('Best epoch: ' + str(epoch))

        state = {
            'net': SiameseNet.module  if args.cuda else SiameseNet,
        }
        torch.save(state, './models1.h5')


#Optimizer Initialization
optimizer = optim.Adam(SiameseNet.parameters(), lr=args.learning_rate, weight_decay = 1e-3)
#Using MSE Loss as it is regression problem
criterion = nn.MSELoss()
if args.cuda:
    SiameseNet = torch.nn.DataParallel(SiameseNet, device_ids=range(torch.cuda.device_count()))
    FinalNet = torch.nn.DataParallel(FinalNet, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
best_loss = 0.999
#Training and validation process
for epoch in range(0,35):
    print('\nEpoch: %d' % epoch)
    train(epoch, SiameseNet, train_loader, optimizer, criterion, args.cuda)
    valid(epoch, SiameseNet, valid_loader, criterion, args.cuda)

#Plots
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
#Saving the model
state = {
        'net': SiameseNet.module if args.cuda else SiameseNet,
        }
torch.save(state, './models.h5')
