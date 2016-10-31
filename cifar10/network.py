import tensorflow as tf
import numpy as np
import config as cf
import functions as F
from tensorflow.python.framework import ops

class BasicConvNet(object):
    def __init__(self, image_w=cf.w, image_h=cf.h, channels=cf.channels, num_classes=10):
        self._width  = image_w # define the width of the image.
        self._height = image_h # define the height of the image.
        self._batch_size = cf.batch_size # define the batch size of mini-batch training.
        self._channels = cf.channels # define the number of channels. ex) RGB = 3, GrayScale = 1, FeatureMap = 50
        self._num_classes = num_classes # define the number of classes for final classfication

        # define the basic options for tensorflow session : restricts allocation of GPU memory.
        gpu_options = tf.GPUOptions(allow_growth = True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # placeholders : None will become the batch size of each batch. The last batch of an epoch may be volatile.
        self._images = tf.placeholder(tf.float32, shape=[None, self._width, self._height, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._is_train = tf.placeholder(tf.bool)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") # saves the global step of training.
        self.UPDATE_OPS_COLLECTION = ops.GraphKeys.UPDATE_OPS

        # loss calculation & update
        self._logits = self._inference(self._images, self._keep_prob, self._is_train) # prediction
        self._avg_loss = self._loss(self._labels, self._logits) # difference between prediction & actual label.
        self._train_op = self._train(self._avg_loss) # back propagate the loss.
        self._accuracy = F.accuracy_score(self._labels, self._logits) # get the accuracy of given prediction batch.

        # basic tensorflow run operations
        self._saver = tf.train.Saver(tf.all_variables())
        self._session.run(tf.initialize_all_variables())

    def prediction(self, X):
        res = None
        for i in range(0, len(X), self._batch_size):
            batch_images = X[i:i+self._batch_size]
            feed_dict = {
                    self._images: batch_images,
                    self._is_train : False,
                    self._keep_prob:1.0
                    }
            test_logits = self._session.run(
                    fetches=self._logits,
                    feed_dict=feed_dict
                    )

            if res is None:
                res = test_logits
            else:
                res = np.r_[res, test_logits]

        return np.argmax(res, axis=1)


    def fit(self, X, y):
        for i in range(0, len(X), self._batch_size): # read whole training dataset by batch size interpolation.
            batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]

            # feeding the dictionary for every placeholder
            feed_dict = {
                    self._images : batch_images,
                    self._labels : batch_labels,
                    self._is_train : True,
                    self._keep_prob : cf.keep_prob
            }

            _, acc, train_avg_loss, global_step = self._session.run(
                    fetches = [
                        self._train_op,
                        self._accuracy,
                        self._avg_loss,
                        self._global_step],
                    feed_dict = feed_dict)
            if (i % cf.display_iter == 0 and i != 0):
                print('Minibatch loss at batch %d: %.5f' %((i/cf.batch_size), train_avg_loss))
                print('Minibatch accuracy: %.2f%%' %(acc*100))

    def score(self, X, y):
        total_acc, total_loss = 0, 0

        tf.get_variable_scope().reuse_variables()
        for i in range(0, len(X), self._batch_size):
            batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]

            feed_dict = {
                    self._images : batch_images,
                    self._labels : batch_labels,
                    self._is_train : False,
                    self._keep_prob : 1.0
            }

            acc, avg_loss = self._session.run(
                    fetches = [
                        self._accuracy,
                        self._avg_loss],
                    feed_dict = feed_dict)

            total_acc += acc * len(batch_images)
            total_loss += avg_loss * len(batch_images)

        return total_acc/len(X), total_loss/len(X)

    def save(self, filepath):
        self._saver.save(self._session, filepath)

    def load(self, filepath):
        self._saver.restore(self._session, filepath)
        print("Model restored.")

    def _inference(self, X, keep_prob, is_train):
        pass

    def _loss(self, labels, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) # No need for one-hot enc.
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.add_to_collection('losses', tf.constant(cf.lr_decay))
        return tf.add_n(tf.get_collection('losses'))

    def _train(self, avg_loss):
        trainable_variables = tf.trainable_variablies()
        grads = tf.gradients(self._loss, trainable_variables)
        optimizer = tf.train.AdamOptimizer().minimize(avg_loss, self._global_step)
        apply_op = optimizer.apply_gradients(zip(grads,trainable_variables),
                global_step=self._global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        return tf.group(*train_ops)

class vggnet(BasicConvNet):
    def _inference(self, X, keep_prob, is_train):
        # Conv_layer 1
        conv = F.conv('conv1', X, 192)
        batch_norm = F._batch_norm(self, 'bn1', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.9, is_train)

        conv = F.conv('conv2', dropout, 192)
        batch_norm = F._batch_norm(self, 'bn2', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.9, is_train)

        max_pool = F.max_pool(dropout) # 16 x 16

        # Conv_layer 2
        conv = F.conv('conv3', max_pool, 192)
        batch_norm = F._batch_norm(self, 'bn3', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.8, is_train)

        conv = F.conv('conv4', dropout, 192)
        batch_norm = F._batch_norm(self, 'bn4', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.8, is_train)

        max_pool = F.max_pool(dropout) # 8 x 8

        # Conv_layer 3
        conv = F.conv('coonv4', max_pool, 256)
        batch_norm = F._batch_norm(self, 'bn5', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.7, is_train)

        conv = F.conv('conv6', dropout, 256)
        batch_norm = F._batch_norm(self, 'bn6', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.7, is_train)

        conv = F.conv('conv7', dropout, 256)
        batch_norm = F._batch_norm(self, 'bn7', conv, is_train)
        dropout = F.dropout(relu, 0.7, is_train)

        max_pool = F.max_pool(dropout) # 4 x 4

         # Conv_layer 4
        conv = F.conv('conv8', max_pool, 512)
        batch_norm = F._batch_norm(self, 'bn8', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.6, is_train)

        conv = F.conv('conv9', dropout, 512)
        batch_norm = F._batch_norm(self, 'bn9', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.6, is_train)

        conv = F.conv('conv10', max_pool, 512)
        batch_norm = F._batch_norm(self, 'bn10', conv, is_train)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.6, is_train)

        max_pool = F.max_pool(dropout) # 2 x 2

        # Fully Connected Layer
        h = tf.reduce_mean(max_pool, reduction_indices=[1,2])
        h = F.dropout(h, 0.5, is_train)
        h = F.dense('fc1', h, 512)
        h = F._batch_norm(self, 'bn11', h, is_train)
        h = F.activation(h)
        h = F.dropout(h, 0.5, is_train)
        h = F.dense('fc2', h, self._num_classes)

        return h

    # Overrriding function _train
    def _train(self, avg_loss):
        lr = tf.select(
                tf.less(self._global_step, cf.step1), 0.1, tf.select(
                    tf.less(self._global_step, cf.step2), 0.02, tf.select(
                        tf.less(self._global_step, cf.step3), 0.004, 0.0008
                        )
                    )
                )

        # moving averages
        variable_averages = tf.train.ExponentialMovingAverage(0.9997, self._global_step)
        tmp_trn_var = tf.trainable_variables()
        update_var = [v for v in tmp_trn_var if v.name != 'global_step:0']
        variable_averages_op = variable_averages.apply(update_var)

        # batch normalizations
        batchnorm_updates = tf.get_collection(self.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        # gradients
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(avg_loss, trainable_variables)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                global_step=self._global_step, name='train_step')

        return tf.group(apply_op, variable_averages_op, batchnorm_updates_op)

class resnet(BasicConvNet):
    def __init__(self, layers, width):
        self._layers = layers
        self._k = width
        super(resnet, self).__init__(image_w=cf.w, image_h=cf.h)

    def _residual(self, h, channels, strides, keep_prob, is_train):
        h0 = h
        h1 = F.conv('conv1', F.activation(F.batch_norm(self, 'bn1', h0, is_train)), channels, strides)
        h1 = F.dropout(h1, keep_prob, is_train)
        h2 = F.conv('conv2', F.activation(F.batch_norm(self, 'bn2', h1, is_train)), channels)
        if F.volume(h0) == F.volume(h2):
            h = h0 + h2
        else :
            h4 = F.conv('conv', h0, channels, strides)
            h = h2 + h4
        return h

    def _inference(self, X, keep_prob, is_train):
        h = F.conv('conv_start', X, 16)
        for i in range(self._layers):
            with tf.variable_scope(str(16*self._k)+'_layers_%s' %i):
                h = self._residual(h, channels=16*self._k, strides=1, keep_prob=keep_prob, is_train=is_train)
        for channels in [32*self._k, 64*self._k]:
            for i in range(self._layers):
                with tf.variable_scope(str(channels)+'_layers_%s' %i):
                    strides = 2 if i == 0 else 1
                    h = self._residual(h, channels, strides, keep_prob, is_train)
        h = F.activation(F.batch_norm(self, 'bn', h, is_train))
        h = tf.reduce_mean(h, reduction_indices=[1,2])
        h = F.dense(h, self._num_classes)

        return h

    def _train(self, avg_loss):
        lr = tf.select(
                tf.less(self._global_step, cf.step1), 0.1, tf.select(
                    tf.less(self._global_step, cf.step2), 0.02, tf.select(
                        tf.less(self._global_step, cf.step3), 0.004, 0.0008
                        )
                    )
                )

        # moving averages
        variable_averages = tf.train.ExponentialMovingAverage(0.95, self._global_step)
        tmp_trn_var = tf.trainable_variables()
        update_var = [v for v in tmp_trn_var if v.name != 'global_step:0']
        variable_averages_op = variable_averages.apply(update_var)

        # batch normalizations
        batchnorm_updates = tf.get_collection(self.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        # gradients
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(avg_loss, trainable_variables)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.95)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                global_step=self._global_step, name='train_step')

        return tf.group(apply_op, variable_averages_op)#, batchnorm_updates_op)

class resnet40(resnet):
    def __init__(self):
        super(resnet40, self).__init__(layers=6, width=1)

class wide_resnet_28x10(resnet):
    def __init__(self):
        super(wide_resnet_28x10, self).__init__(layers=4, width=10)

class resnet28x10(resnet):
    def __init__(self):
        super(resnet28x10, self).__init__(layers=4, width=10)

