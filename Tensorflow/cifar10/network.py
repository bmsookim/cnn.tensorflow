import tensorflow as tf
import numpy as np
import config as cf
import functions as F
from tensorflow.python.framework import ops
from functions import *
from config import *

class BasicConvNet(object):
    def __init__(self, image_w=cf.w, image_h=cf.h, channels=cf.channels, num_classes=10):
        self._width  = image_w # define the width of the image.
        self._height = image_h # define the height of the image.
        self._batch_size = cf.batch_size # define the batch size of mini-batch training.
        self._channels = cf.channels # define the number of channels. ex) RGB = 3, GrayScale = 1, FeatureMap = 50
        self._num_classes = num_classes # define the number of classes for final classfication

        # define the basic options for tensorflow session : restricts allocation of GPU memory.
        gpu_options = tf.GPUOptions(allow_growth = True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # placeholders : None will become the batch size of each batch. The last batch of an epoch may be volatile.
        self._images = tf.placeholder(tf.float32, shape=[None, self._width, self._height, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._is_train = tf.placeholder(tf.bool)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") # saves the global step of training.

        # loss calculation & update
        self._logits = self._inference(self._images, self._keep_prob, self._is_train) # prediction
        self._avg_loss = self._loss(self._labels, self._logits) # difference between prediction & actual label.
        self._train_op = self._train(self._avg_loss) # back propagate the loss.
        self._accuracy = F.accuracy_score(self._labels, self._logits) # get the accuracy of given prediction batch.

        # basic tensorflow run operations
        self._saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)
        self._session.run(tf.initialize_all_variables())

    def prediction(self, X):
        # predicting the label of each image.
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

            # fetching the outputs for session
            _, acc, train_avg_loss, global_step = self._session.run(
                    fetches = [
                        self._train_op,
                        self._accuracy,
                        self._avg_loss,
                        self._global_step],
                    feed_dict = feed_dict)

            # displaying current accuracy & loss
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
        tf.add_to_collection('losses', tf.constant(cf.weight_decay))

        entropy_losses = tf.get_collection('losses')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n(entropy_losses + regularization_losses)
        return loss_
    
    def _average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
        Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _train(self, avg_loss):
        lr = tf.select(
            tf.less(self._global_step, cf.step1), 0.1, tf.select(
                tf.less(self._global_step, cf.step2), 0.02, tf.select(
                    tf.less(self._global_step, cf.step3), 0.004, 0.0008
                    )
                )
            )

        # batch normalizations
        batchnorm_updates = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        # gradients
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        grads = optimizer.compute_gradients(avg_loss)
        apply_op = optimizer.apply_gradients(grads, global_step=self._global_step)

        return tf.group(apply_op, batchnorm_updates_op)

class vggnet(BasicConvNet):
    def _inference(self, X, keep_prob, is_train):
        dropout_rate = [0.9, 0.8, 0.7, 0.6, 0.5]
        layers = [64, 128, 256, 512, 512]
        iters = [2, 2, 3, 3]
        h = X

        # VGG Network Layer
        for i in range(4):
            for j in range(iters[i]):
                with tf.variable_scope('layers%s_%s' %(i, j)) as scope:
                    h = F.conv(h, layers[i])
                    h = F.batch_norm(h, is_train)
                    h = F.activation(h)
                    h = F.dropout(h, dropout_rate[i], is_train)
            h = F.max_pool(h)

        # Fully Connected Layer
        with tf.variable_scope('fully_connected_layer') as scope:
            h = F.dense(h, layers[i+1])
            h = F.batch_norm(h, is_train)
            h = F.activation(h)
            h = F.dropout(h, dropout_rate[i+1], is_train)

        # Softmax Layer
        with tf.variable_scope('softmax_layer') as scope:
            h = F.dense(h, self._num_classes)

        return h

class resnet(BasicConvNet):
    def __init__(self, layers, width):
        self._layers = layers
        self._k = width
        super(resnet, self).__init__(image_w=cf.w, image_h=cf.h)

    def _residual(self, h, channels, strides, keep_prob, is_train):
        h0 = h
        with tf.variable_scope('residual_first'):
            h1 = F.conv(F.activation(F.batch_norm(h0, is_train)), channels, strides)
            h1 = F.dropout(h1, keep_prob, is_train)
        with tf.variable_scope('residual_second'):
            h2 = F.conv(F.activation(F.batch_norm(h1, is_train)), channels)
        if F.volume(h0) == F.volume(h2):
            h = h0 + h2
        else :
            h4 = F.conv(h0, channels, strides)
            h = h2 + h4
        return h

    def _inference(self, X, keep_prob, is_train):
        h = F.conv(X, 16)
        for i in range(self._layers):
            with tf.variable_scope(str(16*self._k)+'layers_%s' %i):
                h = self._residual(h, channels=16*self._k, strides=1, keep_prob=keep_prob, is_train=is_train)
        for channels in [32*self._k, 64*self._k]:
            for i in range(self._layers):
                with tf.variable_scope(str(channels)+'layers_%s' %i):
                    strides = 2 if i == 0 else 1
                    h = self._residual(h, channels, strides, keep_prob, is_train)
        h = F.activation(F.batch_norm(h, is_train))
        h = tf.reduce_mean(h, reduction_indices=[1,2])
        with tf.variable_scope("softmax"):
            h = F.dense(h, self._num_classes)
        return h

class resnet200(resnet):
    def __init__(self):
        super(resnet200, self).__init__(layers=33, width=1)

class wide_resnet_28x10(resnet):
    def __init__(self):
        super(wide_resnet_28x10, self).__init__(layers=4, width=10)

