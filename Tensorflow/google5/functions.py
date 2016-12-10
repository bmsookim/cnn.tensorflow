import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

def dense(x, n):
    W, b = weight_variable([volume(x), n]), bias_variable([n])
    return tf.matmul(x, W) + b

def activation(x):
    # print "ReLU"
    return tf.nn.relu(x)

def max_pool(x, ksize=56, strides=1):
    return tf.nn.max_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='VALID')

def avg_pool(x, ksize=2, strides=2):
    return tf.nn.avg_pool(x, ksize=[1,ksize,ksize,1], strides=[1,strides,strides,1], padding='SAME')

def channels(x):
    return int(x.get_shape()[-1])

def volume(x):
    return np.prod([d for d in x.get_shape()[1:].as_list()])

def flatten(x):
    return tf.reshape(x, [-1, volume(x)])

def dropout(x, ratio):
    return tf.nn.dropout(x, 1.0)

def accuracy_score(labels, logits):
    correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 
    return accuracy

def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))  # mask => 1 for used frame, 0 for unused frame
    length = tf.reduce_sum(used, reduction_indices=1)                 # count of used frame
    length = tf.cast(length, tf.int64)
    return length

def last_relevant_output(output, length):
    # (batch_size, num_step, n_input)
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])

    length = tf.cast(length, tf.int32)
    index = tf.range(0, batch_size) * max_length + (length - 1)

    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

def LSTM_Cell(n_hidden, number_of_layers, keep_prob) :
    lstm_cell = rnn_cell.LSTMCell(n_hidden)
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
    return rnn_cell.MultiRNNCell([lstm_cell] * number_of_layers)

def GRU_Cell(n_hidden, number_of_layers, keep_prob) :
    lstm_cell = rnn_cell.GRUCell(n_hidden)
    lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
    return rnn_cell.MultiRNNCell([lstm_cell] * number_of_layers)

def dynamicRNN(fw_stack, bw_stack, data, sequence_length) : 
    output, _ = rnn.bidirectional_dynamic_rnn(
        fw_stack, bw_stack, 
        data, 
        dtype=tf.float32, 
        sequence_length=sequence_length
    )
    return output

def BiRNN(data, n_hidden, number_of_layers, keep_prob):
    # Actual Length of a sequence to (batch_size) 
    sequence_length = length(data)

    # Get lstm cell input
    # input : data / (batch_size, num_step, n_input)

    # Get lstm cell output
    # outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
    # output_states: A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn.

    # Forward direction cell
    fw_stacked_lstm = LSTM_Cell(n_hidden, number_of_layers, keep_prob)

    # Backward direction cell
    bw_stacked_lstm = LSTM_Cell(n_hidden, number_of_layers, keep_prob)

    outputs = dynamicRNN(fw_stacked_lstm, bw_stacked_lstm, data, sequence_length)

    # (?, 56, 50) / (batch_size, n_steps, 2*n_hidden)
    outputs = tf.concat(2, outputs)
    pooled = tf.reduce_max(outputs, 1) # max_pooling

    # (?, 50) / (batch_size, n_hidden)
    last = last_relevant_output(outputs, sequence_length)

    return last, pooled

def gru_BiRNN(data, n_hidden, number_of_layers, keep_prob):
    # Actual Length of a sequence to (batch_size) 
    sequence_length = length(data)  

    # Get lstm cell input
    # input : data / (batch_size, num_step, n_input)

    # Get lstm cell output
    # outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
    # output_states: A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn.

    # Forward direction cell
    fw_stacked_lstm = GRU_Cell(n_hidden, number_of_layers, keep_prob)

    # Backward direction cell
    bw_stacked_lstm = GRU_Cell(n_hidden, number_of_layers, keep_prob)

    outputs = DynamicRNN(fw_stacked_lstm, bw_stacked_lstm, data, sequence_length)

    # (?, 56, 50) / (batch_size, n_steps, 2*n_hidden)
    outputs = tf.concat(2, outputs)

    # (?, 50) / (batch_size, n_hidden)
    last = last_relevant_output(outputs, sequence_length)


    return last

def prediction(data, n_hidden, number_of_layers, keep_prob, n_class, offset=False, wd=1e-4) :
    weights = tf.Variable(tf.random_normal([2*n_hidden, n_class], stddev=np.sqrt(2.0 / (2*n_hidden*n_class))))
    biases = tf.Variable(tf.random_normal([n_class], stddev=0))

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(weights), wd)
        tf.add_to_collection('losses', weight_decay)
        
    if(offset) : 
        last_output = gru_BiRNN(data, n_hidden, number_of_layers, keep_prob)
    else : 
        last_output, pooled_output = BiRNN(data, n_hidden, number_of_layers, keep_prob)

    output1 = dropout(tf.matmul(last_output, weights) + biases, keep_prob)
    output2 = dropout(tf.matmul(pooled_output, weights) + biases, keep_prob)
    # output3 = dropout(tf.matmul((last_output+pooled_output), weights)+biases, keep_prob)

    return output1
