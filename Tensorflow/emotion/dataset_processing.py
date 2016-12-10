import cv2
import os
import urllib
import numpy as np
import config as cf
import image_processing
from os import path
ROOT = path.dirname(path.dirname(path.abspath(__file__)))
size = cf.size
upscaled_size = cf.upscaled_size

def distort(image, is_train=True):
    image = np.reshape(image, (1, 48, 48))
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(float)
    # if is_train:
        # image = image_processing.upscale(image, (upscaled_size, upscaled_size))
        # image = np.reshape(image, (upscaled_size, upscaled_size, 1))
        # image = image_processing.random_crop(image, (size, size, 1))
        # image = image_processing.random_flip_left_right(image)
        # image = image_processing.random_brightness(image, max_delta=63)
        # image = image_processing.random_contrast(image, lower=0.2, upper=1.8)
    # else:
        # image = image_processing.upscale(image, (size, size))
    image = image_processing.per_image_whitening(image)
    image = np.reshape(image, (size, size, 1))
    return image

def reformat(dataset, image_size, num_channels ):
    '''
     Reshape train/test datasets and labels as tensor type (batch size, image height, image width, channels) 
     for passing them through conv2D built-in function
     
     Input : a tuple of 
             dataset : A 3D numpy array containing input data of shape (batch size, heigth, width)
             labels : A 1D numpy array containing label data of shape (batch size, )
             image_size : Size of image (image shape is square so that widht and heigh has same length)
             num_labels : The number of class (for our case, 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neurtal))
             
     Return : a tuple of
             dataset : A 4D numpy array containing input data of shape(batch size, height, width, num_channels)
             labels : A 2D numpy array containing one hot encoded label data of shape (batch size, num_labels)
    '''
    dataset = np.reshape(dataset, (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset

def load_emotion7(is_train=True):
    path = ROOT + "/input/emotion-pickles/dataset.pickle"
    if os.path.exists(path):
        # test : load pickle file and check dataset 
        emotion7 = image_processing.unpickle(path)
        images, labels = [], []

        if(is_train) : 
            key_data, key_label = 'x_train', 'y_train'
        else :
            key_data, key_label = 'x_test', 'y_test'

        for i in range(len(emotion7[key_label])):
            images.append(distort(emotion7[key_data][i], is_train))
        labels = emotion7[key_label]

        images = np.asarray(images)
        return image_processing.shuffle(images, np.asarray(labels))
    else:
        print(' %s does not exists' % path)
        return