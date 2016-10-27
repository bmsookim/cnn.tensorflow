import cv2
import os
import urllib
import cPickle
import numpy as np
import config as cf
from ZCA import *

def unpickle(filename):
    with open(filename, 'rb') as fp:
        return cPickle.load(fp)

def shuffle(images, labels):
    perm = np.arange(len(labels))
    np.random.shuffle(perm)
    return np.asarray(images)[perm], np.asarray(labels)[perm]

def whitening(image, mode='ZCA'):
    if mode == 'simple' :
        image -= np.mean(image, axis = 0) # zero centered
        image /= np.std(image, aixs = 0)  # normalize
    elif mode == 'ZCA' :
        image = ZCA(image)
    else :
        print ("Wrong augmentation method, no whitening will be occurred")
    return image


def resize(dataset, dim):
    resized_dataset = cv2.resize(dataset, dim, interpolation=cv2.INTER_AREA)
    return resized_dataset

def random_contrast(image, lower=0.2, upper=1.8, seed=None):
    contrast_factor = np.random.uniform(lower, upper)
    return (image-np.mean(image))*contrast_factor + np.mean(image)

def random_brightness(image, max_delta=63, seed=None):
    delta = np.random.randint(-max_delta, max_delta)
    return image-delta

def random_flip_lr(image):
    if np.random.random() < 0.5:
        image = cv2.flip(image,1)
    return image

def random_flip_ud(image):
    if np.random.random() < 0.5:
        image = cv2.flip(image,0)
    return image

def random_crop(image, dim):
    if len(image.shape):
        W, H, D = image.shape
        w, h, d = dim
    else:
        W, H = image.shape
        w, h = size
    left, top = np.random.randint(W-w+1), np.random.randint(H-h+1)
    return image[left:left+w, top:top+h]
