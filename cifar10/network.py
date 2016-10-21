import tensorflow as tf
import numpy as np
import config as cf
import functions as F
from tensorflow.python.training import moving_averages

class BasicConvNet(object):
    def __init__(self, image_w=cf.w, image_h=cf.h, channels=cf.channels, num_classes=10):
        self._width  = image_w # define the width of the image.
        self._height = image_h # define the height of the image.
        self._batch_size = cf.batch_size # define the batch size of mini-batch training.
        self._channels = cf.channels # define the number of channels. ex) RGB = 3, GrayScale = 1, FeatureMap = 50
        self._num_classes = num_classes # define the number of classes for final classfication
        self.mode = cf.mode # define whether to use moving averages
        self._extra_train_ops = [] # extra training ops for moving averages

        # define the basic options for tensorflow session : restricts allocation of GPU memory.
        gpu_options = tf.GPUOptions(allow_growth = True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # placeholders : None will become the batch size of each batch. The last batch of an epoch may be volatile.
        self._images = tf.placeholder(tf.float32, shape=[None, self._width, self._height, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") # saves the global step of training.

        # loss calculation & update
        self._logits = self._inference(self._images, self._keep_prob) # prediction
        self._avg_loss = self._loss(self._labels, self._logits) # difference between prediction & actual label.
        self._train_op = self._train(self._avg_loss) # back propagate the loss.
        self._accuracy = F.accuracy_score(self._labels, self._logits) # get the accuracy of given prediction batch.

        # basic tensorflow run operations
        self._saver = tf.train.Saver(tf.all_variables())
        self._session.run(tf.initialize_all_variables())

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                   initializer=tf.constant_initializer(0.0, tf.float32),trainable=False)
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                   initializer=tf.constant_initializer(1.0, tf.float32),trainable=False)
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                   initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                   initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

                tf.histogram_summary(mean.op.name, mean)
                tf.histogram_summary(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def fit(self, X, y):
        self.mode = 'train'
        for i in range(0, len(X), self._batch_size): # read whole training dataset by batch size interpolation.
            batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]

            # feeding the dictionary for every placeholder
            feed_dict = {
                    self._images : batch_images,
                    self._labels : batch_labels,
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
        self.mode = 'test'
        total_acc, total_loss = 0, 0

        for i in range(0, len(X), self._batch_size):
            batch_images, batch_labels = X[i:i+self._batch_size], y[i:i+self._batch_size]

            feed_dict = {
                    self._images : batch_images,
                    self._labels : batch_labels,
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

    def _inference(self, X, keep_prob):
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
    def _inference(self, X, keep_prob):
        # Conv_layer 1
        conv = F.conv(X, 64)
        batch_norm = self._batch_norm('bn1', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.9, cf.train)

        conv = F.conv(dropout, 64)
        batch_norm = self._batch_norm('bn2', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.9, cf.train)

        max_pool = F.max_pool(dropout) # 16 x 16

        # Conv_layer 2
        conv = F.conv(max_pool, 128)
        batch_norm = self._batch_norm('bn3', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.8, cf.train)

        conv = F.conv(dropout, 128)
        batch_norm = self._batch_norm('bn4', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.8, cf.train)

        max_pool = F.max_pool(dropout) # 8 x 8

        # Conv_layer 3
        conv = F.conv(max_pool, 256)
        batch_norm = self._batch_norm('bn5', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.7, cf.train)

        conv = F.conv(dropout, 256)
        batch_norm = self._batch_norm('bn6', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.7, cf.train)

        conv = F.conv(dropout, 256)
        batch_norm = self._batch_norm('bn7', conv)
        dropout = F.dropout(relu, 0.7, cf.train)

        max_pool = F.max_pool(dropout) # 4 x 4

         # Conv_layer 4
        conv = F.conv(max_pool, 512)
        batch_norm = self._batch_norm('bn8', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.6, cf.train)

        conv = F.conv(dropout, 512)
        batch_norm = self._batch_norm('bn9', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.6, cf.train)

        conv = F.conv(max_pool, 512)
        batch_norm = self._batch_norm('bn10', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.6, cf.train)

        max_pool = F.max_pool(dropout) # 2 x 2

        # Conv_layer 5
        conv = F.conv(max_pool, 512)
        batch_norm = self._batch_norm('bn11', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.5, cf.train)

        conv = F.conv(dropout, 512)
        batch_norm = self._batch_norm('bn12', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.5, cf.train)

        conv = F.conv(max_pool, 512)
        batch_norm = self._batch_norm('bn13', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.5, cf.train)

        max_pool = F.max_pool(dropout) # 1 x 1

        # Fully Connected Layer 1
        fc = F.dense(max_pool, 512)
        batch_norm = self._batch_norm('fc', conv)
        relu = F.activation(batch_norm)
        dropout = F.dropout(relu, 0.5, cf.train)
        h = F.dense(dropout, self._num_classes)

        return h

    # Overriding for function _train
    def _train(self, avg_loss):
        lr = tf.select(
                tf.less(self._global_step, cf.step1), 0.1, tf.select(
                    tf.less(self._global_step, cf.step2), 0.02, tf.select(
                        tf.less(self._global_step, cf.step3), 0.004, 0.0008
                        )
                    )
                )

        # moving averages
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, self._global_step)
        tmp_trn_var = tf.trainable_variables()
        update_var = [v for v in tmp_trn_var if v.name != 'global_step:0']
        variable_averages_op = variable_averages.apply(update_var)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(avg_loss, trainable_variables)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                global_step=self._global_step, name='train_step')
        train_ops = [apply_op]+self._extra_train_ops+[variable_averages_op]
        return tf.group(*train_ops)

class resnet(BasicConvNet):
    def __init__(self, layers, width):
        self._layers = layers
        self._k = width
        super(resnet, self).__init__(image_w=cf.w, image_h=cf.h)

    def _residual(self, h, channels, strides, keep_prob):
        h0 = h
        h1 = F.conv(F.activation(self._batch_norm('bn1', h0)), channels, strides)
        h1 = F.dropout(h1, keep_prob, cf.train)
        h2 = F.conv(F.activation(self._batch_norm('bn2', h1)), channels)
        if F.volume(h0) == F.volume(h2):
            h = h0 + h2
        else :
            h4 = F.conv(h0, channels, strides)
            h = h2 + h4
        return h

    def _inference(self, X, keep_prob):
        h = F.conv(X, 16)
        for i in range(self._layers):
            with tf.variable_scope('16_layer_%s' %i):
                h = self._residual(h, channels=16*self._k, strides=1, keep_prob=keep_prob)
        for channels in [32*self._k, 64*self._k]:
            for i in range(self._layers):
                with tf.variable_scope(str(channels)+'_layer_%s' %i):
                    strides = 2 if i == 0 else 1
                    h = self._residual(h, channels, strides, keep_prob)
        h = F.activation(self._batch_norm('bn3', h))
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
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, self._global_step)
        tmp_trn_var = tf.trainable_variables()
        update_var = [v for v in tmp_trn_var if v.name != 'global_step:0']
        variable_averages_op = variable_averages.apply(update_var)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(avg_loss, trainable_variables)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables),
                global_step=self._global_step, name='train_step')
        train_ops = [apply_op]+self._extra_train_ops+[variable_averages_op]
        return tf.group(*train_ops)

class resnet202(resnet):
    def __init__(self):
        super(resnet202, self).__init__(layers=33, width=1)

class resnet28x10(resnet):
    def __init__(self):
        super(resnet28x10, self).__init__(layers=4, width=10) # layer should be (depth-4)/6


