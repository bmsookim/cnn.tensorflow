import os
import time, datetime
import argparse
import numpy as np
import pandas as pd
import network
from network import *
import dataset_processing as datasets

resume = False
train  = True

def run(clf, train):
    test_images, test_labels = datasets.load_emotion7(is_train=False)
    save_dir = "../models/%s" % clf.__class__.__name__
    records = []

    if(train) :
        # if resume mode is on, load the checkpoint file
        if(resume) :
            try :
                clf.load(save_dir + "/model.ckpt")
                print "Resumed from" + str(save_dir + "/model.ckpt") + "\n"
            except ValueError :
                print "Starting a new model" + str(save_dir + "/model.ckpt") + "\n"

        # number of epochs and the time
        start_time = time.time()
        save_epoch = -1
        exploded = False

        for epoch in range(cf.epochs):
            if(exploded) :
                print "Restoring previous epoch..."
                epoch = save_epoch+1
                exploded = False
            print "Epoch #%d" % (epoch+1)
            epoch_time = time.time()
            # load new sets of training images
            train_images, train_labels = datasets.load_emotion7(is_train=True)

            clf.fit(train_images, train_labels, max_epoch=1)

            duration = time.time() - start_time
            time_per = time.time() - epoch_time
            test_accuracy, test_loss = clf.score(test_images, test_labels)
            test_accuracy *= 100

            summary = {
                "epoch": epoch,
                "name": clf.__class__.__name__,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "time_per": time_per,
            }

            print('\nTest accuracy: %.2f%%' % test_accuracy)
            print('Test loss: %.6f' % test_loss)
            print "Elapsed Time: {:.2f}".format(duration/60), "Minutes"
            
            records.append(summary)
            df = pd.DataFrame(records)
            df.to_csv("../output/%s.csv" % clf.__class__.__name__.lower(), index=False)
            if df["test_accuracy"].max() - 1e-5 < test_accuracy and test_loss < 2:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                print("Save to %s" % save_dir)
                clf.save(save_dir + "/model.ckpt")
            if test_loss > 5 :
                clf.load(save_dir + "/model.ckpt")
                exploded = True
            print ""

    clf.load(save_dir + "/model.ckpt")
    test_accuracy, test_loss = clf.score(test_images, test_labels)
    test_accuracy *= 100

    print('\nTest accuracy: %.2f%%\n' % test_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default=cf.classifier, choices=dir(network))
    args = parser.parse_args()
    run(eval("%s()" % args.class_name), train)
