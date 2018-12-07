# -*- coding: utf-8 -*-
"""
简介
CS20-使用placeholder方式建立计算图--logreg
Created on Fri Dec  7 11:38:30 2018

@author: gear
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import time
import utils

tf.reset_default_graph()
# define parameters for the model
learning_rate=0.01
batch_size = 128
n_epochs = 30

# step read in data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)


# create placeholders for X, Y
X = tf.placeholder(tf.float32, shape=[784, batch_size], name='image')
Y = tf.placeholder(tf.float32, shape=[10, batch_size], name='label')

# step3 create weights and bias
# shape of w depends on the dimension of X and Y
# shape of b depends on Y
w = tf.get_variable(name='weight', shape=[10, 784], initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=[10, 1], initializer=tf.zeros_initializer())

# step4 build the model
logits = tf.matmul(w, X) + b

# step5 create the loss function

logits = tf.transpose(logits)
labels = tf.transpose(Y)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='loss')
loss = tf.reduce_mean(entropy)

# step6 using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# step7 calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
#tf.cast(x, dtype, name=None),将x的数据格式转化成dtype
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg_placeholder', tf.get_default_graph())

# step8 train the model
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)
    
    for i in range(n_epochs):
        total_loss = 0
        
        for j in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            X_batch = np.transpose(X_batch)
            Y_batch = np.transpose(Y_batch)
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
            total_loss += batch_loss
            
        print('Average loss epochs {0}: {1}'.format(i, total_loss / n_batches))
    end = time.time()
    print('Total time: {0} seconds'.format(end - start))
    
    # step9 test the model
    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0
    
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        X_batch = np.transpose(X_batch)
        Y_batch = np.transpose(Y_batch)
        accuracy_batch = sess.run(accuracy, feed_dict={X:X_batch, Y:Y_batch})
        total_correct_preds += accuracy_batch
        
    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
    
writer.close()


