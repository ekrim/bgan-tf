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

def prepare_noisy_labels(label, batch_size):
  if label == 1:
    y = np.ones((batch_size, 1), dtype=np.float32)
    idx = np.random.choice(batch_size, int(0.05*batch_size), replace=False)
    y[idx,0] +=  

def noise_std_scheduler(cnt):
  if cnt < 20000:
    return 0.001 + 0.05*cnt/20000
  else:
    return 0.001

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


def train(batch_size=256, epochs=10, dim_z=128, lr=0.001, buffer_sz=32768):
  dim_h = 128
  img_dim = (64, 64, 3)

  # placeholders
  image_ph = tf.placeholder(tf.float32, (None,) + img_dim) 
  z_ph = tf.placeholder(tf.float32, (None, dim_z))
  D_target_ph = tf.placeholder(tf.float32, (None, 1))

  # training related
  training_ph = tf.placeholder(tf.bool, ())
  noise_ph = tf.placeholder(tf.float32, ())
  
  # build the input image iterator
  image_tens, iterator = CelebAInput().input_fn(
    mode='train', 
    batch_size=batch_size,
    epochs=epochs)

  # discriminator, generator, and generator score
  with tf.variable_scope('models') as scope:
    D_out = discriminator(image_ph, noise_ph, training_ph, dim_h=dim_h)
    G_image = generator(z_ph, training_ph, dim_h=dim_h)
    scope.reuse_variables()
    D_fake = discriminator(G_image, noise_ph, training_ph, dim_h=dim_h)

  # prepare losses
  D_loss = -tf.reduce_mean(
    D_target_ph*tf.log(D_out) + (1-D_target_ph)*tf.log(1-D_out))

  G_loss = 0.5 * tf.reduce_mean(tf.square(
    tf.log(D_fake) - tf.log(1-D_fake)))

  # ignore D variables when training G
  vars_to_train_G = np.setdiff1d(
    tf.trainable_variables(),
    tf.trainable_variables('models/discriminator'))

  # train ops
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    D_train = tf.train.SGD(learning_rate=lr).minimize(D_loss)
    G_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(
      G_loss, var_list=vars_to_train_G)

  # buffer of generated images
  old_images = ImageBuffer(buffer_sz=buffer_sz)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    cnt = 0 
    while False:
      try:
        label = 1 if cnt%2==0 else 0
        if label == 1:
          image_batch = sess.run(image_tens)
	else:
	  image_batch = sess.run(
	    G_image, 
	    feed_dict={
	      z_ph: np.random.randn(batch_size, *img_dim).astype(np.float32),
	      training_ph: False})

	sess.run(
	  D_train, 
	  feed_dict={
	    image_ph: image_batch,
	    D_target_ph: prepare_noisy_labels(label, batch_size),
            noise_ph: noise_std_scheduler(cnt), 
	    training_ph: True})

	cnt += 1
      except tf.errors.OutOfRangeError:
        break


if __name__=='__main__':

  train(batch_size=2048, epochs=1, dim_z=128, buffer_sz=32768)
