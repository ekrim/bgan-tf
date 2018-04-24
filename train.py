from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf

from data_pipeline import CelebAInput
from models import generator, \
                   discriminator
 

HEIGHT = 64
WIDTH = 64
CHANNELS = 3


def bgan_class():
  _loss
  pass


class ImageBuffer:
  '''Generated images are added to this buffer
  The discriminator trains on a mix of freshly erated images
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


def train(batch_size=256, epochs=10, dim_z=128, buffer_sz=32768):
  dim_h = 128
  # placeholders
  z_ph = tf.placeholder(tf.float32, (None, dim_z))
  D_target_ph = tf.placeholder(tf.float32, (None, 1))
  training_ph = tf.placeholder(tf.bool, ())
  
  # build the input image iterator
  image_tens, iterator = CelebAInput().input_fn(
    mode='train', 
    batch_size=batch_size,
    epochs=epochs)

  # discriminator, generator, and generator score
  with tf.variable_scope('models') as scope:
    D_logits = discriminator(image_tens, training_ph, dim_h=dim_h)
    G_image = generator(z_ph, training_ph, dim_h=dim_h)
    scope.reuse_variables()
    D_fake = discriminator(G_image, training_ph, dim_h=dim_h)

  # prepare losses
  #D_loss = tf.nn.softplu-
  G_loss = tf.reduce_mean(tf.square(D_fake))

  # buffer of erated images
  old_images = ImageBuffer(buffer_sz=buffer_sz)

  vars_to_train_dis = np.setdiff1d(
    tf.trainable_variables(),
    tf.trainable_variables('models/generator'))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    cnt = 0 
    while False:
      try:
        #x = sess.run(D_logits, feed_dict={training_ph:False})
        #x = sess.run(, feed_dict={z_ph: np.random.randn(batch_size, dim_z).astype(np.float32), training_ph:False})
        print(x.shape)
        cnt += x.shape[0]
        print(cnt)
      except tf.errors.OutOfRangeError:
        break
    print(cnt)


if __name__=='__main__':

  train(batch_size=2048, epochs=1, dim_z=128, buffer_sz=32768)
