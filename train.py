from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf
 

HEIGHT = 64
WIDTH = 64
CHANNELS = 3


class PredictionBuffer:
  '''Generated images are added to this buffer
  The discriminator trains on a mix of freshly generated images
  and images from this buffer
  '''
  def __init__(self, buffer_sz):
    self.buffer_sz = buffer_sz
    self.buffer = np.zeros((buffer_sz, HEIGHT, WIDTH, CHANNELS), dtype=np.float32)
    self.cnt = 0 
  
  def add(self, x):
    if self.cnt < self.buffer_sz:
      self.buffer[self.cnt:min(self.cnt+x.shape[0], self.buffer_sz)] = x
    else:
       
      

  def sample(self, batch_size):
    idx = np.random.choice(self.cnt, batch_size, replace=True)
    return self.buffer[idx]


def train(epochs=10, dis_buffer=32768):
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
