# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
multiply = tf.multiply(a, b)

with tf.Session() as sess:
	print('a+b=' , sess.run(add, feed_dict={a: 2, b: 3}))
	print('a*b=' , sess.run(multiply, feed_dict={a: 2, b: 3}))