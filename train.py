from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf

from models import generator, \
                   discriminator
 

HEIGHT = 64
WIDTH = 64
CHANNELS = 3


class ImageBuffer:
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
      n_to_add = min(self.buffer_sz-self.cnt, x.shape[0])
      self.buffer[self.cnt:self.cnt+n_to_add] = x[:n_to_add]
      self.cnt += n_to_add
    else:
      idx = np.random.choice(self.buffer_sz, x.shape[0], replace=False).sort()
      self.buffer[idx] = x

  def sample(self, batch_size):
    idx = np.random.choice(self.cnt, batch_size, replace=True)
    return self.buffer[idx]


def train(epochs=10, dim_z=64, buffer_sz=32768):
  z_ph = tf.placeholder(tf.float32, (None, dim_z))
  dis_target_ph = tf.placeholder(tf.float32, (None, 1))
  training_ph = tf.placeholder(tf.bool, ())
  
  input_fn = CelebAInput().input_fn_factory(
    mode='train', 
    batch_size=2)

  image_tens = input_fn()  
  dis_logits = discriminator(image_tens, training_ph, dim_h=dim_h)
  gen_image = generator(z_ph, training_ph, dim_h=dim_h)

  old_images = ImageBuffer(buffer_sz=buffer_sz)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    cnt = 0
    for i_epoch in range(epochs):
       
      sess.run()
      print(sess.run(output).shape)
      print(cnt)
      cnt += 1
