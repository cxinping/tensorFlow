# -*- coding: utf-8 -*-


import tensorflow as tf

x = tf.Variable(3)
y = tf.Variable(5)
z=x+y
init = tf.global_variables_initializer()

with tf.Session() as sess: 
    sess.run(init) 
    print(sess.run(z))