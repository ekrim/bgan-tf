from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf
 

def train(epochs=10):
  x = tf.placeholder(tf.float32, (None, 218, 178, 3))
  y = tf.placeholder(tf.float32, (None, 10))
  training_ph = tf.placeholder(tf.bool, ())
  
  input_fn = CelebAInput().input_fn_factory(
    mode='train', 
    batch_size=2)

  image = input_fn()  
  output = model_test(image, mode='train')

  tf.Session(
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #for i_epoch in range(epochs):
    cnt = 0
    while True:
      print(sess.run(output).shape)
      print(cnt)
      cnt += 1
