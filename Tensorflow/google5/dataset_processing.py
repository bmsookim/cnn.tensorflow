import numpy as np
import cPickle
import pandas as pd
from os import path
ROOT = path.dirname(path.dirname(path.abspath(__file__)))

def get_idx_from_sent(sent, word_idx_map, max_l):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    #words = sent.split()
    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, word_idx_map_amr, num_class, max_l, max_l_amr):
    """
    Transforms sentences into a 2-d matrix.
    """
    trainX, trainX_amr, trainY = [], [], []
    valX, valX_amr, valY = [], [], []
    testX, testX_amr, testY = [], [], []

    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l)
        nodes = get_idx_from_sent(rev["amrNodes"], word_idx_map_amr, max_l_amr)
        label = rev["y"]

        # test = split 2
        if rev["split"]==2:
            testX.append(sent)
            testX_amr.append(nodes)
            testY.append(label)

        # validation = split 3
        elif rev["split"]==3:
            valX.append(sent)
            valX_amr.append(nodes)
            valY.append(label)
            # trainX.append(sent)
            # trainX_amr.append(nodes)
            # trainY.append(label)

        # train = split 1
        else:
            trainX.append(sent)
            trainX_amr.append(nodes)
            trainY.append(label)

    trainX = np.array(trainX, dtype=int)
    trainX_amr = np.array(trainX_amr, dtype=int)
    trainY = np.array(trainY, dtype=int)

    valX = np.array(valX, dtype=int)
    valX_amr = np.array(valX_amr, dtype=int)
    valY = np.array(valY, dtype=int)

    testX = np.array(testX, dtype=int)
    testX_amr = np.array(testX_amr, dtype=int)
    testY = np.array(testY, dtype=int)
    return [trainX, trainX_amr, trainY, valX, valX_amr, valY, testX, testX_amr, testY]

def make_vector_data(dataset, W):
    """
    word index to word vectors
    """
    new_dataX = []
    for sentence in dataset:
        new_sentence = []
        for token_idx in sentence:
            new_sentence.append(W[token_idx])
        new_dataX.append(new_sentence)
    new_dataX = np.array(new_dataX)

    return new_dataX

def loadPickleData():
    print "Loading pickle data...",
    x = cPickle.load(open(ROOT+"/input/sst-glove/sst-glove.p","rb"))
    revs, W, word_idx_map, W_amr, word_idx_map_amr, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    print "Complete!!"

    max_l = np.max(pd.DataFrame(revs)["len"])
    max_l_amr = np.max(pd.DataFrame(revs)["amrNodes_len"])
    # print "Max Lengths:", max_l, max_l_amr
    
    """
    Convert sentence to word index, further it should be converted to vectors with W
    """
    num_class = 5
    datasets = make_idx_data_cv(revs, word_idx_map, word_idx_map_amr, num_class, max_l, max_l_amr)

    trainX = datasets[0]
    trainX_amr = datasets[1]
    trainY = datasets[2]

    valX = datasets[3]
    valX_amr = datasets[4]
    valY = datasets[5]

    testX = datasets[6]
    testX_amr = datasets[7]
    testY = datasets[8]

    """
    Convert word index to word vectors
    """
    new_trainX = make_vector_data(trainX, W)
    new_trainX_amr = make_vector_data(trainX_amr, W_amr)
    # (8534, 96, 300) (8534, 5)

    new_valX = make_vector_data(valX, W)
    new_valX_amr = make_vector_data(valX_amr, W_amr)
    # (1099, 96, 300) (1099, 5)

    new_testX = make_vector_data(testX, W)
    new_testX_amr = make_vector_data(testX_amr, W_amr)
    # (2208, 96, 300) (2208, 5)

    isAMR = 1
    if isAMR == 1:
        # print "AMR Mode"
        return new_trainX, new_trainX_amr, trainY, new_valX, new_valX_amr, valY, new_testX, new_testX_amr, testY
        #new_trainX = np.concatenate( (new_trainX, new_trainX_amr), axis=1 )
        #new_valX = np.concatenate( (new_valX, new_valX_amr), axis=1 )
        #new_testX = np.concatenate( (new_testX, new_testX_amr), axis=1 )

    else:
        return new_trainX, trainY, new_valX, valY, new_testX, testY

def shuffle(inputs, labels):
    perm = np.arange(len(labels))
    np.random.shuffle(perm)
    return np.asarray(inputs)[perm], np.asarray(labels)[perm]