# -*- coding: utf-8 -*-
"""
简介：
CS20课程代码：
lecture1: Overview of Tensorflow

Created on Tue Dec  4 21:10:04 2018
@author: dell
"""

import tensorflow as tf

# 创建一个简单的计算图
a = tf.add(3, 5)
print(a)          # 这样不会显示

# 显示计算结果
with tf.Session() as sess:
    print(sess.run(a))
    
# tensorflow使用默认的计算图，若自己创建计算图：
g = tf.Graph()

# 将自己创建的计算图设置为默认计算图
with g.as_default():
    x = tf.add(3, 5)  # 这样创建的x在计算图g中
    
# 计算这个变量的结果
#with tf.Session() as sess:  # 这样会出错，因为这样使用默认的计算图
#    print(sess.run(x))  
    
# 正确的结果如下
with tf.Session(graph=g) as sess:
    print(sess.run(x))
    
# 得到默认的计算图 
g1 = tf.get_default_graph()
g2 = tf.Graph()

# 将tensor加入默认计算图
a = tf.constant(3)
with tf.Session(graph=g1) as sess:
    print(sess.run(a))

# 将tensor加入自己的计算图
with g2.as_default():
    b = tf.constant(4)
with tf.Session(graph=g2) as sess:
    print(sess.run(b))