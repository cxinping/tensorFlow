# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.examples.tutorials.mnist  

#1
print('\n#1,set.dat')
rlog= r'E:\quant\WinPythonbi36\log_tf\iris'
a = tf.constant(1.0, name='ta')
b = tf.constant(2.0, name='tb')
c=a+b

#2
print('\n#2,Session')
ss = tf.Session()

#3
print('\n#3,Session.run')
xss=ss.run(c)
print('xss,',xss)

#4
print('\n#4,summary')
xsum= tf.summary.FileWriter(rlog, ss.graph)  
print('rlog',rlog)

#5
print('\n#5,Session.close')
ss.close()






