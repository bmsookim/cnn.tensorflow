import os
import time, datetime
import argparse
import numpy as np
import pandas as pd
import network
from network import *
import batch_load as datasets
import config as cf

def run(clf):
    test_images, test_labels = datasets.load_cifar100(is_train=False)
    records = []
    save_dir = "../models/" + cf.dataset# + ("/%s" % clf.__class__.__name__)

    clf.load(save_dir + "/" + cf.model + ".ckpt")
    test_accuracy, test_loss = clf.score(test_images, test_labels)
    test_accuracy *= 100

    print('\nTest accuracy: %.2f%%\n' % test_accuracy)

    return test_images, clf.prediction(test_images), test_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default=cf.model, choices=dir(network))
    args = parser.parse_args()
    run(eval("%s()" % args.class_name))
