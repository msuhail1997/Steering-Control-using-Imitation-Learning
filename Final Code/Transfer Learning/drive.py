#Headers to be imported
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
import torchvision.transforms as transforms
import utils

#Server Initialization
sio = socketio.Server()
app = Flask(__name__)
#Initial model and image Initialization
model = None
prev_image_array = None

#Minimum and Maximum Spped Limit
MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED
transformations = transforms.Compose([transforms.ToPILImage()])
transformations1 = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x/127.5 - 1)])

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


#Event Handler
@sio.on('telemetry')
#Telemetry Function
def telemetry(sid, data):
    if data:
        # The current steering angle from environment
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])

        # The current image perceived
        original_image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            # from PIL image to numpy array
            image = np.asarray(original_image)
            # Teh preprocessing needed to be applied
            image = utils.preprocess(image)
            image = transformations1(transforms.functional.resize(transformations(image),size=(224,224)))
            # image = torch.Tensor(image)
            #image = np.array([image])       # the model expects 4D array

            image = image.view(1, 3, 224, 224)
            image = Variable(image)

            # Model to predict the Sttering Angle
            steering_angle = model(image).view(-1).data.numpy()[0]
            print(type(steering_angle))

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print("Exception")
            print(e)

        # saves frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            original_image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


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
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
