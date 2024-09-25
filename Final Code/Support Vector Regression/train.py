# NON DEEP LEARNING BASELINE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import argparse
import os
from skimage.color import lab2rgb, lch2lab

from model import *
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage.color
from sklearn.svm import SVR
import pickle
import matplotlib.pyplot as plt

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


#
class Grayscale(BaseEstimator, TransformerMixin):
    """
    Class to convert images to grayscale
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return np.array([skimage.color.rgb2gray(img) for img in X])

#

class TransformHOG(BaseEstimator, TransformerMixin):
    """
    class to apply HOG transformation
    """
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])



def load_data(args):
    """
    load data
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    data_df = data_df[data_df['left'] != 'left']
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0, shuffle=True)

    return X_train, X_valid, y_train, y_valid


X_train, X_valid, y_train, y_valid = load_data(args)
transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])
train_set = TrainCarDataset4Images(X_train, y_train, args.data_dir,transformations)
valid_set = ValidCarDataset(X_valid, y_valid, args.data_dir,transformations)


gray = Grayscale()
hogimage = TransformHOG(
    pixels_per_cell=(8, 8),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)

loss_graph=[]
x_axis=[]


def SupportVectorRegression():
    """
    machine learning model
    """
    train_loss = 0
    X_train=[]
    Y_train=[]
    count=0
    loss_values=[]
    for (centers, lefts, rights, center_flips) in train_set:
        centers, lefts, rights, center_flips = centers,lefts,rights,center_flips
        datas = [lefts, rights, centers, center_flips]
        for data in datas:
            imgs, targets = data
            X_train.append(imgs)
            Y_train.append(targets)
            count=count+1
            print(count)
        #working with only 4000 images
        if(count>4000):
            break
    print("Hog Transform processing.....")
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_train_gray = gray.fit_transform(X_train)
    X_train_hog = hogimage.fit_transform(X_train_gray)

    """
    plotting loss and steering angle graphs
    """
    clf = SVR(C=1.0, epsilon=0.2)
    for i in range(0,X_train_hog.shape[0],100):
        clf.fit(X_train_hog[:i+1], Y_train[:i+1])
        Y_pred=clf.predict(X_train_hog[:i+1])
        loss_values = ((Y_pred-Y_train[:i+1])**2).mean()
        print("Loss",loss_values)
        loss_graph.append(loss_values)
        x_axis.append(i)
    clf.fit(X_train_hog, Y_train)
    Y_pred=clf.predict(X_train_hog)
    Y_train=list(Y_train)
    Y_pred=list(Y_pred)

    filename="modeltrack2.sav"
    pickle.dump(clf,open(filename,"wb"))
    print("Model Saved")


    plt.plot(Y_train[0::10])
    plt.plot(Y_pred[0::10])
    plt.legend(['Ground Steering angle','Predicted Steering angle'])
    plt.title("SVR (Non DL Baseline) Track 2 Steering angle: Training graph")
    plt.xlabel("Number of Data Points")
    plt.ylabel("Steering angle")
    plt.show()
    plt.plot(x_axis,loss_graph)
    plt.title("Support Vector Regression (Non DL Baseline) Track 2")
    plt.xlabel("Number of Data Points")
    plt.ylabel("Training Loss")
    plt.show()

SupportVectorRegression()
