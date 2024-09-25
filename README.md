# Steering Control using Imitation Learning


1) Dataset url: https://www.kaggle.com/zaynena/selfdriving-car-simulator

The "dataset" directory contains two sub-directories- "track1data" and "track2data" corresponding to datasets of two diffferent terrains.

2) Simulator url: https://github.com/udacity/self-driving-car-sim
(Version 2 simulator was used)


*******************************************

The models implemented were:
* Support Vector Regression- ML baseline model
* CNN- DL baseline model
* MTL (Multi-Task Learning) for Speed Control
* Few-shot learning 
* Transfer Learning

A comparitive study was done between these 5 models.
Each model has a README.md file to run that module.

model.py - model file
train.py - training and validation code
utils.py - helper functions
drive.py - driver code to run the model on the simulator
