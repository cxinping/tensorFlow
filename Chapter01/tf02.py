# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

x=np.array([[1,1,1],[1,-8,1],[1,1,1]])
w=tf.Variable(initial_value=x)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))