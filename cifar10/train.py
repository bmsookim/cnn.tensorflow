import os
import time, datetime
import argparse
import numpy as np
import pandas as pd
import network
from network import *
import batch_load as datasets

resume = False
train = cf.train

def run(clf):
    test_images, test_labels = datasets.load_cifar10(is_train=False)
    records = []
    save_dir = "../models/" + cf.dataset# + ("/%s" % clf.__class__.__name__)

    # if resume mode is on, load the checkpoint file
    if(resume) :
        try :
            clf.load(save_dir + "/"+ cf.model + ".ckpt")
            print "Resumed from" + str(save_dir + "/model.ckpt") + "\n"
        except ValueError :
            print "Starting a new model" + str(save_dir + "/model.ckpt") + "\n"

    # number of epochs and the time
    start_time = time.time()
    save_epoch = 0
    exploded = False

    for epoch in range(save_epoch, cf.epochs):
        if(exploded) :
            print "Restoring previous epoch..."
            epoch = save_epoch+1
            exploded = False
        print "Epoch #%d" % (epoch+1)
        epoch_time = time.time()

        # load new sets of training images
        train_images, train_labels = datasets.load_cifar10(is_train=True)

        clf.fit(train_images, train_labels)

        duration = time.time() - start_time
        time_per = time.time() - epoch_time
        clf.mode = 'validation'
        test_accuracy, test_loss = clf.score(test_images, test_labels)
        test_accuracy *= 100

        summary = {
            "epoch": (epoch+1),
            "name": clf.__class__.__name__,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "time_per": time_per,
        }

        print('\nBest model test accuracy: %.2f%%' % test_accuracy)
        print('Test loss: %.5f' % test_loss)
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed Time: {:0>2}h {:0>2}m {:0>2}s".format(int(hours),int(minutes),int(seconds)))

        records.append(summary)
        df = pd.DataFrame(records)
        if not os.path.exists("../output/"+cf.dataset+"/"):
            os.mkdir("../output/"+cf.dataset+"/")
        df.to_csv("../output/"+cf.dataset+("/%s.csv" % clf.__class__.__name__.lower()), index=False)
        if df["test_accuracy"].max() - 1e-5 < test_accuracy and test_loss < 200.0 :
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            print(("Save to %s" % save_dir)+"/"+cf.model+".ckpt")
            clf.save(save_dir +"/"+ cf.model + ".ckpt")
            save_epoch = epoch
        if test_loss > 5 :
            clf.load(save_dir+"/"+ cf.model + ".ckpt")
            exploded = True
        print ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default=cf.model, choices=dir(network))
    args = parser.parse_args()
    run(eval("%s()" % args.class_name))
