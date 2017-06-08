# -*- coding: utf-8 -*-


import tensorflow as tf

word=tf.constant('hello,world!')
with tf.Session() as sess:
    print(sess.run(word))
    
    