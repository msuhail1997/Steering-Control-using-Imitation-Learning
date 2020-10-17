#preprocessing

import cv2, os
import numpy as np
import matplotlib.image as mpimg
import torch.utils.data as data

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 16, 32, 1


def load_image(direc, image_file):
    f = os.path.join(direc, image_file.strip())
    return mpimg.imread(f)


def preprocess(image):
    """
    All preprocessing functiona in one
    """
    image = image[60:-25, :, :] # remove the sky and the car front
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


def flip(image):
    image = cv2.flip(image, 1)
    return image

def make_random_changes(image, sa):
	"""
    Flip the images randomly
    """
	if np.random.rand() < 0.5:
		image = flip(image)
		sa = -sa

	"""
	Do random translation opertaions
	"""
	h, w = image.shape[:2]
	new_x = 100 * (np.random.rand() - 0.5)
	new_y = 10 * (np.random.rand() - 0.5)
	sa += new_x * 0.002
	trans_m = np.float32([[1, 0, new_x], [0, 1, new_y]])
	image = cv2.warpAffine(image, trans_m, (w, h))

	"""
	Add Shadows randomly
	"""
	x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
	x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
	xmask, ymask = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

	mask = np.zeros_like(image[:, :, 1])
	mask[np.where((ymask - y1) * (x2 - x1) - (y2 - y1) * (xmask - x1) > 0)] = 1
	rand = np.random.uniform(low=0.2, high=0.5)
	condition = mask == np.random.randint(2)

	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	hls[:, :, 1][condition] = hls[:, :, 1][condition] * rand
	image =  cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


	"""
	Change brightness of the images
	"""
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
	hsv[:,:,2] =  hsv[:,:,2] * ratio
	image =  cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

	return image, sa


def generalize_image(image, sa):
    if np.random.rand() < 0.6:
    	image, sa = make_random_changes(image, sa)
    image = preprocess(image)
    return image, sa

class ValidCarDataset(data.Dataset):

    def __init__(self, X, y, direc, transform=None):

        self.X = X
        self.y = y
        self.direc = direc
        self.transform = transform

    def __getitem__(self, index):
        center, left, right = self.X[index]
        sa = self.y[index]
        image, sa = load_image(self.direc, center), float(sa)
        image = preprocess(image)

        if self.transform is not None:
            image = self.transform(image)
        return (image, sa)

    def __len__(self):
        return self.X.shape[0]

class TrainCarDataset4Images(data.Dataset):

    def __init__(self, X, y, direc, transform=None):

        self.X = X
        self.y = y
        self.direc = direc
        self.transform = transform

    def __getitem__(self, index):
        center, left, right = self.X[index]
        sa = self.y[index]

        image_left, sa_left = generalize_image(load_image(self.direc, left), float(sa) + 0.2)

        image_center, sa_center = generalize_image(load_image(self.direc, center), float(sa))

        image_center_flip , sa_center_flip = generalize_image(flip(load_image(self.direc, center)), -1*float(sa))

        image_right, sa_right = generalize_image(load_image(self.direc, right), float(sa) - 0.2)

        if self.transform is not None:
            image_left = self.transform(image_left)
            image_center = self.transform(image_center)
            image_right = self.transform(image_right)

        return (image_center, sa_center), (image_left, sa_left), (image_right, sa_right), (image_center_flip, sa_center_flip)

    def __len__(self):
        return self.X.shape[0]
