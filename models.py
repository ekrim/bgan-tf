from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf
 

def make_bn(training_ph):
  def bn(x):
    return tf.layers.batch_normalization(x, axis=-1, training=training_ph)
  return bn


def discriminator_bgan(images, noise_std_ph, training_ph, dim_h=128):
  '''Implemented https://github.com/rdevon/BGAN/blob/master/models/dcgan_64_pub.py in TensorFlow
    
  NOTE: with batchnorm, put the train op within controlled
  dependencies for update ops
  '''
  bn = make_bn(training_ph)

  def conv_lrelu(x, ch_out):
    x = tf.layers.conv2d(
      inputs=x,
      filters=ch_out,
      kernel_size=(5,5),
      strides=(2,2),
      padding='same')

    return tf.nn.leaky_relu(x, alpha=0.2)

  with tf.variable_scope('discriminator'):
    x = tf.add(images, 
      tf.random_normal(tf.shape(images), mean=0.0, stddev=noise_std_ph))
    x = conv_lrelu(x, dim_h)
    x = bn(conv_lrelu(x, dim_h*2))
    x = bn(conv_lrelu(x, dim_h*4))
    x = bn(conv_lrelu(x, dim_h*8))
    x = tf.reshape(x, [-1, dim_h*8*4*4])
    logits = tf.layers.dense(inputs=x, activation=None, units=1)
    output = tf.nn.sigmoid(logits)

  return output


def generator_bgan(input_z, training_ph, dim_h=128):
  '''Implemented https://github.com/rdevon/BGAN/blob/master/models/dcgan_64_pub.py in TensorFlow
    
  NOTE: with batchnorm, put the train op within controlled
  dependencies for update ops
  '''
  bn = make_bn(training_ph)

  with tf.variable_scope('generator'):
    x = bn(tf.layers.dense(inputs=input_z, units=dim_h*8*4*4))
    x = tf.reshape(x, [-1, 4, 4, dim_h*8])
    x = bn(tf.layers.conv2d_transpose(x, dim_h*4, 5, strides=2, padding='same'))
    x = bn(tf.layers.conv2d_transpose(x, dim_h*2, 5, strides=2, padding='same'))
    x = bn(tf.layers.conv2d_transpose(x, dim_h, 5, strides=2, padding='same'))
    x = tf.layers.conv2d_transpose(x, 3, 5, strides=2, padding='same')
    x = tf.nn.tanh(x)

  return x


def generator_dcgan(input_z, training_ph, dim_h=100):
  '''generator as described in the original DCGAN paper
  https://arxiv.org/pdf/1511.06434.pdf
  '''
  bn = make_bn(training_ph)

  with tf.variable_scope('generator'):
    x = bn(tf.layers.dense(inputs=input_z, units=4*4*1024))
    x = tf.reshape(x, [-1, 4, 4, 1024])
    x = bn(tf.layers.conv2d_transpose(x, 512, 5, strides=2, padding='same'))


if __name__=="__main__":
  training_ph = tf.placeholder(tf.bool, ())
  x_dis = tf.placeholder(tf.float32, (None, 64, 64, 3))
  x_gen = tf.placeholder(tf.float32, (None, 64))
  noise_ph = tf.placeholder(tf.float32, ())
  y_dis = discriminator_bgan(x_dis, noise_ph, training_ph, 10)
  y_gen = generator_bgan(x_gen, training_ph, 4)  

  with tf.Session() as sess:
    tf.summary.FileWriter('./', sess.graph).close()

    sess.run(tf.global_variables_initializer())
    x1 = np.random.randn(2,64,64,3).astype(np.float32)
    x2 = np.random.randn(2,64).astype(np.float32)
    y1 = sess.run(y_dis, feed_dict={x_dis:x1, training_ph:False, noise_ph:0.05}) 
    y2 = sess.run(y_gen, feed_dict={x_gen:x2, training_ph:False}) 
    print('discriminator output shape:')
    print(y1.shape)
    print('generator output shape:')
    print(y2.shape)
