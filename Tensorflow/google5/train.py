import os
import time, datetime
import argparse
import numpy as np
import pandas as pd
import network
from network import *
import tensorflow as tf
import dataset_processing as dataset

batch_x_all, batch_x_all_amr, batch_y_all, X_val, X_val_amr, Y_val, X_test, X_test_amr, Y_test = dataset.loadPickleData()

resume = True
train = True

def run(clf, train):
    records = []
    save_dir = "../models/%s" % clf.__class__.__name__

    if(train) :
        # if resume mode is on, load the checkpoint file
        if(resume) :
            clf.load(save_dir + "/model.ckpt")

        # number of epochs and the time
        start_time = time.time()
        save_epoch = -1
        exploded = False
        for epoch in range(cf.epochs):
            epoch_time = time.time()
            if(exploded) :
                epoch = save_epoch+1
                exploded = False

            print "Epoch #%d" % (epoch+1)
            # load new sets of training images
            X, Y = dataset.shuffle(batch_x_all, batch_y_all)
            clf.fit(X, batch_x_all_amr, Y, max_epoch=1)

            duration = time.time() - start_time
            time_per = time.time() - epoch_time
            test_accuracy, test_loss = clf.score(X_test, X_test_amr, Y_test)
            test_accuracy *= 100

            summary = {
                "epoch": (epoch+1),
                "name": clf.__class__.__name__,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "time_per": time_per,
            }

            print('\nTest accuracy: %.2f%%' % test_accuracy)
            print('Test loss: %.5f' % test_loss)
            print "Elapsed Time: {:.2f}".format(duration/60), "Minutes"
            
            records.append(summary)
            df = pd.DataFrame(records)
            df.to_csv("../output/%s.csv" % clf.__class__.__name__.lower(), index=False)
            if df["test_accuracy"].max() - 1e-5 < test_accuracy and test_accuracy > 47.96 :
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                print("Save to %s" % save_dir)
                clf.save(save_dir + "/model.ckpt")
                save_epoch = epoch
                print save_epoch
            if test_loss > 1.4 :
                clf.load(save_dir + "/model.ckpt")
                exploded = True
            print ""

    clf.load(save_dir + "/model.ckpt")
    test_accuracy, test_loss = clf.score(X_test, X_test_amr, Y_test)
    test_accuracy *= 100

    print('\nTest accuracy: %.2f%%\n' % test_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default=cf.classifier, choices=dir(network))
    args = parser.parse_args()
    run(eval("%s()" % args.class_name), train)