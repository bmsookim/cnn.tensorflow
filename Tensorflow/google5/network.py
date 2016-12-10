import numpy as np
import tensorflow as tf
import functions as F
import config as cf
from sklearn.metrics import accuracy_score

class BaseSentimentClassifier(object):
    def __init__(self, embedding_dim=cf.embedding_dim, time_steps=56, AMR_steps = 40, num_classes=5, batch_size=cf.batch_size):
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._hidden_size = cf.hidden_size

        self._embedding_dim = embedding_dim
        self._time_steps = time_steps
        self._amr_steps = AMR_steps

        gpu_options = tf.GPUOptions(allow_growth =True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self._input1 = tf.placeholder(tf.float32, shape=[None, self._time_steps, self._embedding_dim])
        self._input2 = tf.placeholder(tf.float32, shape=[None, self._amr_steps, self._embedding_dim])
        self._labels = tf.placeholder(tf.int64, shape=[None])

        self._keep_prob = tf.placeholder(tf.float32)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") 
        self._logits = self._inference(self._input1, self._input2, self._keep_prob)
        self._avg_loss = self._loss(self._labels, self._logits)
        self._train_op = self._train(self._avg_loss)

        self._accuracy = F.accuracy_score(self._labels, self._logits)
        self._saver = tf.train.Saver(tf.all_variables())
        self._session.run(tf.initialize_all_variables())

    def fit(self, X, X_amr, y, max_epoch = 10):
        for epoch in range(max_epoch):
            for i in range(0, len(X), self._batch_size):
                x1, x2, batch_labels = X[i:i+self._batch_size], X_amr[i:i+self._batch_size], y[i:i+self._batch_size]
                feed_dict={self._input1: x1, self._input2: x2, self._labels: batch_labels, self._keep_prob: 0.7}
                _, acc, train_avg_loss, global_step = self._session.run(fetches=[self._train_op, self._accuracy, self._avg_loss, self._global_step], feed_dict=feed_dict)

                if(i % cf.display_iter == 0 and i != 0) :
                    print('Minibatch loss at batch %d: %f' % ((i/self._batch_size), train_avg_loss))
                    print('Minibatch accuracy: %.2f%%' % (acc*100))
                    # print('Global step: %d' % global_step)

    def score(self, X, X_amr, y):
        total_acc, total_loss = 0, 0
        for i in range(0, len(X), self._batch_size):
            x1, x2, batch_labels = X[i:i+self._batch_size], X_amr[i:i+self._batch_size], y[i:i+self._batch_size]
            feed_dict={self._input1: x1, self._input2: x2, self._labels: batch_labels, self._keep_prob: 1.0}
            acc, avg_loss = self._session.run(fetches=[self._accuracy, self._avg_loss], feed_dict=feed_dict)
            total_acc += acc * len(x1)
            total_loss += avg_loss * len(x1)
        return total_acc / len(X), total_loss / len(X)

    def save(self, filepath):
        self._saver.save(self._session, filepath)

    def load(self, filepath):
        self._saver.restore(self._session, filepath)
        print ("Model restored.")

    def _inference(self, X, keep_prob):
        pass

    def _loss(self, labels, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean) 
        return tf.add_n(tf.get_collection('losses'))

    def _train(self, avg_loss):
        return tf.train.AdamOptimizer().minimize(avg_loss, self._global_step)

class SentimentClassifier_LSTM(BaseSentimentClassifier):
    def __init__(self):
        super(SentimentClassifier_LSTM, self).__init__()

    def _inference(self, X1, X2, keep_prob):
        pred1 = F.prediction(X1, self._hidden_size, 2, keep_prob, self._num_classes)
        # pred2 = F.prediction(X2, self._hidden_size, 2, keep_prob, self._num_classes, True)
        
        logits = pred1

        return logits
   
    def _train(self, avg_loss):
        lr = tf.select(tf.less(self._global_step, 3200), 0.05, \
            tf.select(tf.less(self._global_step, 4800), 0.04, 0.03))
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum = 0.9).minimize(avg_loss, global_step=self._global_step)

class SentimentClassifier_ResLSTM(BaseSentimentClassifier):
    def __init__(self):
        super(SentimentClassifier_ResLSTM, self).__init__()

    def _inference(self, X1, X2, keep_prob):
        pred1 = F.prediction(X1, self._hidden_size, cf.num_layers, keep_prob, self._num_classes)
        # pred2 = F.prediction(X2, self._hidden_size, 2, keep_prob, self._num_classes, True)
        
        logits = pred1

        return logits
   
    def _train(self, avg_loss):
        lr = tf.select(tf.less(self._global_step, 3200), 0.05, \
            tf.select(tf.less(self._global_step, 4800), 0.045, 0.04))
        return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(avg_loss, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N, global_step=self._global_step)