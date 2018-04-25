from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil
import glob
import numpy as np
import tensorflow as tf

from data_pipeline import CelebAInput
from utils import ChkptSwap, \
                  ImageBuffer, \
                  plot_images
from models import generator, \
                   discriminator
 

INFO = {
  'img_dim': (64, 64, 3),
  'dim_h': 128,
  'batch_size': 256,
  'epochs': 30,
  'lr_G': 0.0001,
  'lr_D': 0.0001}


def prepare_noisy_labels(label, batch_size):
  y = np.asarray([label]*batch_size).astype(np.float32)[:,None]
  y += (1-2*label)*0.3*np.random.rand(batch_size, 1)
  idx = np.random.choice(batch_size, int(0.05*batch_size), replace=False)
  y[idx,0] = 1 - y[idx,0]
  return y.astype(np.float32)


def noise_std_scheduler(cnt):
  if cnt < 20000:
    return 0.001 + 0.05*cnt/20000
  else:
    return 0.001


def train(batch_size=256, epochs=10, dim_z=128, lr_D=0.0001, lr_G=0.0001, buffer_sz=32768, **kwargs):

  if os.path.exists('img'):
    shutil.rmtree('img')
  os.mkdir('img')

  dim_h = kwargs['dim_h']
  img_dim = kwargs['img_dim']

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
  D_vars = tf.trainable_variables('models/discriminator')
  G_vars = tf.trainable_variables('models/generator')
  D_names = [var.name for var in D_vars]
  vars_to_train_G = [var for var in tf.trainable_variables() if var.name not in D_names]

  # train ops
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    D_train = tf.train.GradientDescentOptimizer(learning_rate=lr_D).minimize(D_loss)
    G_train = tf.train.AdamOptimizer(learning_rate=lr_G).minimize(
      G_loss, var_list=vars_to_train_G)

  # buffer of generated images
  image_buffer = ImageBuffer(img_dim=img_dim, buffer_sz=buffer_sz)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    # checkpoint swappers
    D_swap = ChkptSwap(D_vars, 'D') 
    G_swap = ChkptSwap(G_vars, 'G') 

    cnt = 0 
    D_loss_av, G_loss_av = 0.0, 0.0
    while True:
      try:
        label = 1 if cnt%2==0 else 0
        if label == 1:
          image_batch = sess.run(image_tens)
        else:
          image_batch = sess.run(
            G_image,
            feed_dict={
              z_ph: np.random.randn(batch_size, dim_z).astype(np.float32),
              training_ph: False})

          if cnt > 100:
            old_images = image_buffer.sample(batch_size//4)
            image_batch[:batch_size//4] = old_images

        _, D_loss_batch = sess.run(
          [D_train, D_loss],
          feed_dict={
            image_ph: image_batch,
            D_target_ph: prepare_noisy_labels(label, batch_size),
            noise_ph: noise_std_scheduler(cnt), 
            z_ph: np.zeros((batch_size, dim_z), dtype=np.float32),
            training_ph: True})

        _, G_loss_batch, G_image_batch = sess.run(
          [G_train, G_loss, G_image],
          feed_dict={
            image_ph: np.zeros((batch_size,)+img_dim).astype(np.float32),
            z_ph: np.random.randn(batch_size, dim_h).astype(np.float32),
            noise_ph: noise_std_scheduler(cnt),
            training_ph: True})
       
        D_loss_av += D_loss_batch
        G_loss_av += G_loss_batch

        image_buffer.add(G_image_batch)

        if cnt > 50:
          D_swap.step()
          G_swap.step()      

        if cnt%10==0:
          D_loss_av /= 10
          G_loss_av /= 10
          print('step {:d} -- G loss: {:0.4f} -- D_loss: {:0.4f}'.format(
            cnt,
            G_loss_av,
            D_loss_av))
          D_loss_av, G_loss_av = 0.0, 0.0 
 
          if cnt%1000==0:
            plot_images(G_image_batch[:16], 'img/img{:08d}.png'.format(cnt))

        cnt += 1

      except tf.errors.OutOfRangeError:
        break


if __name__=='__main__':
  train(**INFO)
