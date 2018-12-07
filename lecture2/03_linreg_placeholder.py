# -*- coding: utf-8 -*-
"""
简介
CS20-使用placeholder方式建立计算图--linreg
Created on Fri Dec  7 10:20:36 2018

@author: gear
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils


DATA_FILE = 'data/birth_life_2010.txt'

# step1 read in data from file
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# step2 create palceholders for X and Y
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=None, name='X')
Y= tf.placeholder(tf.float32, shape=None, name='Y')

# step3 create weight and bias, initialized to 0
w = tf.get_variable(name='weight', shape=None, initializer=tf.constant(0.0))
b= tf.get_variable(name='bias', shape=None, initializer=tf.constant(0.0))

# step4 build model to predict Y
Y_predicted = w * X + b

# step5 create loss function --MSE(mean squareed error)
loss = tf.square(Y - Y_predicted, name='loss')

# step6 using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# step7 compute the Graph
start = time.time()

writer = tf.summary.FileWriter('./graphs/linear_reg', tf.get_default_graph())


with tf.Session() as sess:
    # initialize the variable, in this case is w and b
    sess.run(tf.global_variables_initializer())
    
    # train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, cost = sess.run([optimizer, loss], feed_dict={X:x, Y:y})
            total_loss += cost
            
            print('Epoch {0}:{1}'.format(i, total_loss/n_samples))
            
    # close the writer  when you're done using it 
    writer.close()
        
    # step 8 output the values of w ana b
    w_out, b_out = sess.run([w, b])
    
    end = time.time()
    print('Took: %f seconds'%(end - start))
    

# step9 plot the results

plt.plot(data[:,0], data[:,1], 'bo', label='real_data')
plt.plot(data[:,0], w_out * data[:,0] + b_out, 'r', label='pred_data')
plt.legend()
plt.show()
    
    

