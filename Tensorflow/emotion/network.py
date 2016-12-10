import numpy as np
import tensorflow as tf
import functions as F
import config as cf
from sklearn.metrics import accuracy_score

class BaseED7Classifier(object):
    def __init__(self, image_size=cf.size, num_classes=7, batch_size=50, channels=1):
        self._image_size = image_size
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._channels = channels
        gpu_options = tf.GPUOptions(allow_growth =True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._images = tf.placeholder(tf.float32, shape=[None, self._image_size, self._image_size, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") 
        self._logits = self._inference(self._images, self._keep_prob)
        self._avg_loss = self._loss(self._labels, self._logits)
        self._train_op = self._train(self._avg_loss)
        self._accuracy = F.accuracy_score(self._labels, self._logits)
        self._saver = tf.train.Saver(tf.all_variables())
        self._session.run(tf.initialize_all_variables())

    def fit(self, X, y, max_epoch = 10):
        for epoch in range(max_epoch):
            for i in range(0, len(X), self._batch_size):
                batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]
                feed_dict={self._images: batch_images, self._labels: batch_labels, self._keep_prob: 0.7}
                _, acc, train_avg_loss, global_step = self._session.run(fetches=[self._train_op, self._accuracy, self._avg_loss, self._global_step], feed_dict=feed_dict)

                if(i % cf.display_iter == 0 and i != 0) :
                    print('Minibatch loss at batch %d: %f' % ((i/self._batch_size), train_avg_loss))
                    print('Minibatch accuracy: %.2f%%' % (acc*100))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        res = None
        for i in range(0, len(X), self._batch_size):
            batch_images = X[i:i+self._batch_size]
            feed_dict={self._images: batch_images, self._keep_prob: 1.0}
            test_logits = self._session.run(fetches=self._logits, feed_dict=feed_dict)
            if res is None:
                res = test_logits
            else:
                res = np.r_[res, test_logits]
        return res

    def score(self, X, y):
        total_acc, total_loss = 0, 0
        for i in range(0, len(X), self._batch_size):
            batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]
            feed_dict={self._images: batch_images, self._labels: batch_labels, self._keep_prob: 1.0}
            acc, avg_loss = self._session.run(fetches=[self._accuracy, self._avg_loss], feed_dict=feed_dict)
            total_acc += acc * len(batch_images)
            total_loss += avg_loss * len(batch_images)
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

class ED7Classifier_ResNet(BaseED7Classifier):
    def __init__(self, layers, k):
            self._layers = layers
            self._k = k
            super(ED7Classifier_ResNet, self).__init__(image_size=cf.size, batch_size=cf.batch_size)

    def _residual(self, h, channels, strides, keep_prob):
        h0 = h
        h1 = F.dropout(F.conv(F.activation(F.batch_normalization(h0)), channels, strides), keep_prob)
        h2 = F.conv(F.activation(F.batch_normalization(h1)), channels)
        # c.f. http://gitxiv.com/comments/7rffyqcPLirEEsmpX
        if F.volume(h0) == F.volume(h2):
            h = h2 + h0
        else:
            h4 = F.conv(h0, channels, strides)
            h = h2 + h4
        return h

    def _inference(self, X, keep_prob):
        h = X
        h = F.conv(h, 16)
        for i in range(self._layers):
            h = self._residual(h, channels=16*self._k, strides=1, keep_prob=keep_prob)
        for channels in [32*self._k, 64*self._k]:
            for i in range(self._layers):        
                strides = 2 if i == 0 else 1
                h = self._residual(h, channels, strides, keep_prob)
        h = F.activation(F.batch_normalization(h))
        h = tf.reduce_mean(h, reduction_indices=[1, 2]) # Global Average Pooling
        h = F.dense(h, self._num_classes)
        return h
   
    def _train(self, avg_loss):
        lr = tf.select(tf.less(self._global_step, cf.step1), 0.1, \
            tf.select(tf.less(self._global_step, cf.step2), 0.02, \
            tf.select(tf.less(self._global_step, cf.step3), 0.004, 0.0008)))
        # return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(avg_loss, \
        return tf.train.AdamOptimizer().minimize(avg_loss, \
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N, global_step=self._global_step)

class ED7Classifier_WRN16x8(ED7Classifier_ResNet):
    def __init__(self):
        super(ED7Classifier_WRN16x8, self).__init__(layers=2, k=8)

class ED7Classifier_ResNet16(ED7Classifier_ResNet):
    def __init__(self):
        super(ED7Classifier_ResNet16, self).__init__(layers=2, k=2)


