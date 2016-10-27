import os
import numpy as np
import config as cf
import augmentation as aug
import os.path as path

ROOT = path.dirname(path.dirname(path.abspath(__file__)))

def augment(image, is_train=True):
    image = np.reshape(image, (3,cf.w,cf.h))
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

def load_cifar10(is_train):
    if is_train:
        filenames = [ROOT+"/input/cifar-10-batches-py/data_batch_%d" % j for j in range(1,6)]
    else:
        filenames = [ROOT+"/input/cifar-10-batches-py/test_batch"]

    images, labels = [], []
    for filename in filenames:
        dictionary = aug.unpickle(filename) # import pickle file as a dictionary
        for i in range(len(dictionary["labels"])):
            images.append(augment(dictionary["data"][i], is_train))
        labels += dictionary["labels"]

    if is_train:
        return aug.shuffle(images, np.asarray(labels))
    else:
        return images, np.asarray(labels)
