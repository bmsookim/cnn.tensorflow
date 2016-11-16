import os
import numpy as np
import config as cf
import augmentation as aug
import os.path as path
import matplotlib.pyplot as plt

ROOT = path.dirname(path.dirname(path.abspath(__file__)))

def augment(image, is_train=True):
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1,2,0))
    image = image.astype(float)

    if is_train:
        image = aug.resize(image, (cf.uw, cf.uh))
        image = aug.random_crop(image, (cf.w, cf.h, 3))
        image = aug.random_flip_lr(image)
        image = aug.random_brightness(image)
        image = aug.random_contrast(image)
    else:
        image = aug.resize(image, (cf.w, cf.h))
    return aug.whitening(image)

def load_cifar100(is_train):
    if is_train:
        filenames = [ROOT+"/input/cifar-100-python/train"]
    else:
        filenames = [ROOT+"/input/cifar-100-python/test"]

    images, labels = [], []
    for filename in filenames:
        dictionary = aug.unpickle(filename) # import pickle file as a dictionary
        for i in range(len(dictionary["fine_labels"])):
            images.append(augment(dictionary["data"][i], is_train))
        labels += dictionary["fine_labels"]

    if is_train:
        return aug.shuffle(images, np.asarray(labels))
    else:
        return images, np.asarray(labels)
