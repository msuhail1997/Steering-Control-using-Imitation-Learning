Few-Shot Learning:
Please make the following directory changes in train.py:
--Change the path of the three images by downloading the images 'center_2019_04_02_18_05_57_110.jpg','center_2019_04_02_19_25_53_860.jpg', 'center_2019_04_02_19_28_27_725.jpg' in the function SiameseNetworkDataset.
--Change the path of the images by downloading the images from the Support set folder in the function SupportSet.
--Change the path of the two directories: data_dir to track2data(Jungle terrain) and data_dir2 to to track1data(Plains terrain). Download the data using following link: https://www.kaggle.com/zaynena/selfdriving-car-simulator


In order to run the training loop, run the following command ' python3 train.py'
After the training process, in order to run the model on the simulator(can be downloaded from https://github.com/udacity/self-driving-car-sim), run the following command 'python3 drive.py MODELNAME'. Subsequently, run the simulator in Autonomous Mode. The drive.py file communicates with the simulator to run the saved model.
