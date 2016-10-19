import cv2
import os
import urllib
import cPickle
import numpy as np
# import matplotlib.pyplot as plt

# def img_plot(img, normalize=True):
#     if normalize:
#         img_max, img_min = np.max(img), np.min(img)
#         img = 255.0*(img-img-min)/(img_max-img_min)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')

def unpickle(filename):
    with open(filename, 'rb') as fp:
        return cPickle.load(fp)

def shuffle(images, labels):
    perm = np.arange(len(labels))
    np.random.shuffle(perm)
    return np.asarray(images)[perm], np.asarray(labels)[perm]

def whitening(image):
    return (image-np.mean(image))/np.std(image)

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
