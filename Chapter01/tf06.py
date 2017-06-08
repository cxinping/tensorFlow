# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.Variable(tf.ones([3,2]))
b = tf.Variable(tf.ones([2,3]))
#product = tf.matmul(5*a,4*b)
product= tf.matmul(tf.multiply(5.0,a),tf.multiply(4.0,b))
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print(sess.run(product))