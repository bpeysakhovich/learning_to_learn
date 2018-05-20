import numpy as np
import tensorflow as tf
from parameters import par
import pickle
import task
import os


# TODO: move these parameters to a better home
filters = [32,32,64,64]
kernel_size = [3, 3]
pool_size = [2,2]
stride = 1
num_layers = len(filters)
dense_layers = [4096, 2000, 1000] # 4096 is the size after the convolutional layers
training_iterations = 40000
learning_rate = 0.001
image_dataset = 'imagenet'

use_gpu = True
if use_gpu:
    gpu_id = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id



def train_weights_image_plus_spatial_classification():
    # need to train convolutional weights to classify images
    # and to infer the spatial location of (colored) saccade targets
    # saccade target locations can be inferred from 1st and/or 2nd convolutional layers
    pass

def train_weights_image_classification():

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    input_data   = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'input')
    target_data  = tf.placeholder(tf.float32, [par['batch_size'], dense_layers[-1]], 'target')

    # pass input through convolutional layers
    x = apply_convulational_layers(input_data, None)

    # pass input through dense layers
    with tf.variable_scope('dense_layers'):
        c = 0.1
        W0 = tf.get_variable('W0', initializer = tf.random_uniform([dense_layers[0], dense_layers[1]], -c, c), trainable = True)
        W1 = tf.get_variable('W1', initializer = tf.random_uniform([dense_layers[1], dense_layers[2]], -c, c), trainable = True)
        b0 = tf.get_variable('b0', initializer = tf.zeros([1, dense_layers[1]]), trainable = True)
        b1 = tf.get_variable('b1', initializer = tf.zeros([1, dense_layers[2]]), trainable = True)

    x = tf.nn.relu(tf.matmul(x, W0) + b0)
    y = tf.matmul(x, W1) + b1

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = target_data, dim = 1))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss)

    # we will train the network on imagenet dataset
    stim = task.Stimulus(image_dataset)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(training_iterations):

            batch_data, batch_labels = stim.generate_image_batch()
            _, train_loss  = sess.run([train_op, loss], feed_dict = {input_data: batch_data, target_data: batch_labels})

            if i%1000 == 0:
                print('Iteration ', i, ' Loss ', train_loss)

        W = {}
        for var in tf.trainable_variables():
            W[var.op.name] = var.eval()

        pickle.dump(W, open(par['conv_weight_fn'],'wb'))
        print('Convolutional weights saved in ', par['conv_weight_fn'])


def apply_convulational_layers(x, saved_weights_file):

    # load previous weights is saved_weights_file is not None,
    # otherwise, train new weights
    if saved_weights_file is None:
        kernel_init = [None for _ in range(num_layers)]
        bias_init = [tf.zeros_initializer() for _ in range(num_layers)]
        train = True
    else:
        kernel_names = ['conv2d']
        for i in range(1, num_layers):
            kernel_names.append('conv2d_' + str(i))
        conv_weights = pickle.load(open(saved_weights_file,'rb'))
        kernel_init = [tf.constant_initializer(conv_weights[k + '/kernel']) for k in kernel_names]
        bias_init = [tf.constant_initializer(conv_weights[k + '/bias']) for k in kernel_names]
        train = False

    for i in range(num_layers):

        x = tf.layers.conv2d(inputs = x, filters = filters[i], kernel_size = kernel_size, kernel_initializer = kernel_init[i],  \
            bias_initializer = bias_init[i], strides = stride, activation = tf.nn.relu, padding = 'SAME', trainable = train)

        if i > 0 and i%2 == 1:
            # apply max pooling and dropout after every second layer
            x = tf.layers.max_pooling2d(inputs = x, pool_size = pool_size, strides = 2, padding='SAME')
            x = tf.nn.dropout(x, par['drop_keep_pct'])

    return tf.reshape(x, [par['batch_size'], -1])
