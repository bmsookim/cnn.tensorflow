import os
import time, datetime
import argparse
import numpy as np
import pandas as pd
import network
from network import *
import batch_load as datasets

def run(clf, train):
    test_images, test_labels = datasets.load_cifar10(is_train=False)
    records = []
    save_dir = "../models/" + cf.dataset# + ("/%s" % clf.__class__.__name__)
    
    clf.load(save_dir + "/" + cf.dataset + ".ckpt")
    test_accuracy, test_loss = clf.score(test_images, test_labels)
    test_accuracy *= 100

    print('\nTest accuracy: %.2f%%\n' % test_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default=cf.model, choices=dir(network))
    args = parser.parse_args()
    run(eval("%s()" % args.class_name), train)