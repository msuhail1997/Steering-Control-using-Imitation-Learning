#Header Files
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
import torch
from torch.autograd import Variable
from model import *
from utils import crop , rgb2hsv, preprocess
import torchvision.transforms as transforms
import utils
import matplotlib.image as mpimg
#Server Initialization
sio = socketio.Server()
app = Flask(__name__)
#Initialization of the Model as empty
model = None
prev_image_array = None
#Minimum and Maximum speed for the vehicle
MAX_SPEED = 25
MIN_SPEED = 10
transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])
class SiameseNetworkDataset():

    def __init__(self,transform=None,should_invert=True):
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self):
        img0path = '/home/raghav/Documents/Spring2020/DLDS/time/center_2019_04_02_18_05_57_110.jpg'
        img0 =  mpimg.imread(img0path)
        global steering_img0
        steering_img0 = 0

        # img1 = Image.open('/home/raghav/Documents/track2data/IMG/left_2019_04_02_18_08_28_950.jpg')
        img1path = '/home/raghav/Documents/Spring2020/DLDS/time/center_2019_04_02_19_25_53_860.jpg'
        img1 =  mpimg.imread(img1path)
        global steering_img1
        #steering_img1 = 0.7500002 + 0.2
        #steering_img1 = 0.7500002
        #steering_img1 = -0.6000001
        steering_img1 = -0.2000001

        # img2 = Image.open('/home/raghav/Documents/track2data/IMG/right_2019_04_02_18_06_53_687.jpg')
        img2path = '/home/raghav/Documents/Spring2020/DLDS/time/center_2019_04_02_19_28_27_725.jpg'
        global steering_img2
        #45steering_img2 = -0.95 - 0.2
        #steering_img2 = 0.6500001
        steering_img2 = 0.2500001

        img2 =  mpimg.imread(img2path)
        img0 = preprocess(img0)
        img1 = preprocess(img1)
        img2 = preprocess(img2)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2


siamese_compare = SiameseNetworkDataset(transforms.ToTensor())
#Passing the fixed image list for training and validation 
def returnImage():
    img0,img1,img2 = siamese_compare.__getitem__()
    image_list = [steering_img0, steering_img1, steering_img2]
    image_list = torch.FloatTensor(image_list)
    img0_list = img0.repeat(1,1,1,1)
    img1_list = img1.repeat(1,1,1,1)
    img2_list = img2.repeat(1,1,1,1)
    image_list = image_list.repeat(1,1)
    print("IMage list",image_list)
    #Support list

    return img0_list, img1_list, img2_list,image_list

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)
img0_list, img1_list, img2_list,image_list = returnImage()

#Event Handler for the server
@sio.on('telemetry')
#Telemetry
def telemetry(sid, data):
    if data:
        # Current Steering Angle
        steering_angle_previous = float(data["steering_angle"])
        #Current Throttle
        throttle = float(data["throttle"])
        #Speed
        speed = float(data["speed"])
        #Current Image
        original_image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            #Converting PIL Image to numpy
            image = np.asarray(original_image)
            #Apply Preprocessing
            image = utils.preprocess(image)
            image = transformations(image)
            image = torch.Tensor(image)
            #Reshaping to the image size
            image = image.view(1, 3, 75, 320)
            image = Variable(image.float())
            #Pass the images to the model
            steering_angle = model(image,img0_list,img1_list,img2_list,image_list).view(-1).data.numpy()[0]
            # lower the throttle as the speed increases
            # Speed > Speed Limit -----> Downhill
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            print('{}'.format(steering_angle))
            #Sending the variables to the simulator
            send_control(steering_angle , throttle)
        except Exception as e:
            print("Exception")
            print(e)

        #Saving the frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            original_image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
#Connecting to the server
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

#Sending variables to the simulator
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    model = checkpoint['net']
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("Record ...")
    else:
        print("Not Recording now ...")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
