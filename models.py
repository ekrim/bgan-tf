from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf
 
from data_pipeline import CelebAInput

def train(epochs=10):
  x = tf.placeholder(tf.float32, (None, 218, 178, 3))
  y = tf.placeholder(tf.float32, (None, 10))
  #loss = model_test()
  
  input_fn = CelebAInput().input_fn_factory(
    mode='train', 
    batch_size=2)

  image = input_fn()  
  output = model_test(image, mode='train')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #for i_epoch in range(epochs):
    cnt = 0
    while True:
      print(sess.run(output).shape)
      print(cnt)
      cnt += 1
     

def discriminator(features, training_ph, dim_h):
  '''Implemented https://github.com/rdevon/BGAN/blob/master/models/dcgan_64_pub.py in TensorFlow
    
  NOTE: with batchnorm, put the train op within controlled
  dependencies for update ops
  '''
  x = tf.layers.conv2d(
    inputs=features['image'],
    filters=dim_h,
    kernel_size=(5,5),
    strides=(2,2),
    padding='same',
    activation=None)

  x = tf.nn.leaky_relu(x, alpha=0.01)

  x = tf.layers.conv2d(
    inputs=x,
    filters=dim_h*2,
    kernel_size=(5,5),
    strides=(2,2),
    padding='same',
    activation=None)

  x = tf.nn.leaky_relu(x, alpha=0.01)

  x = tf.layers.batch_normalization(
    x,
    axis=-1,
    training=training_ph)

  x = tf.layers.conv2d(
    inputs=x,
    filters=dim_h*4,
    kernel_size=(5,5),
    strides=(2,2),
    padding='same',
    activation=None)

  x = tf.nn.leaky_relu(x, alpha=0.01)

  x = tf.layers.batch_normalization(
    x,
    axis=-1,
    training=training_ph)

  x = tf.layers.conv2d(
    inputs=x,
    filters=dim_h*8,
    kernel_size=(5,5),
    strides=(2,2),
    padding='same',
    activation=None)

  x = tf.nn.leaky_relu(x, alpha=0.01)

  x = tf.layers.batch_normalization(
    x,
    axis=-1,
    training=training_ph)

  x = tf.reshape(x, [-1, dim_h*8*4*4])
    
  logits = tf.layers.dense(inputs=x, activation=None, units=1)

  return logits


def generator(input_z, training_ph, dim_h, dim_z):
  '''Implemented https://github.com/rdevon/BGAN/blob/master/models/dcgan_64_pub.py in TensorFlow
    
  NOTE: with batchnorm, put the train op within controlled
  dependencies for update ops
  '''

  x = tf.layers.dense(
    inputs=input_z, 
    activation=None, 
    units=dim_h*8*4*4)

  x = tf.layers.batch_normalization(
    x,
    axis=-1,
    training=training_ph)

  x = tf.reshape(x, [-1, 4, 4, dim_h*8])

  x = tf.nn.conv2d_transpose(
    x,
    (
    layer = batch_norm(Deconv2DLayer(
            layer, dim_h * 4, 5, stride=2, pad=2))

  x = tf.layers.batch_normalization(
    x,
    axis=-1,
    training=training_ph)

  x = tf.nn.leaky_relu(x, alpha=0.01)

  x = tf.layers.conv2d(
    inputs=x,
    filters=dim_h*2,
    kernel_size=(5,5),
    strides=(2,2),
    padding='same',
    activation=None)

  x = tf.nn.leaky_relu(x, alpha=0.01)

  x = tf.layers.batch_normalization(
    x,
    axis=-1,
    training=training_ph)

def model_fn_closure(model_name='test'):
  '''model_name is one of "test" or "all_cnn"
  '''
  if model_name == 'test':
    inference_fn = model_test
  elif model_name == 'all_cnn':
    inference_fn = model_all_cnn_c  
  else:
    assert False, 'model not implemented'

  def model_fn(features, labels, mode, params):
    '''Model function for estimators
    '''
    logits = inference_fn(features, mode)
  
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
  
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions) 
     
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

      train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

      training_hooks = [tf.train.SummarySaverHook(
        save_steps=50,
        summary_op=tf.summary.merge_all())]        

      return tf.estimator.EstimatorSpec(
        mode=mode, 
        loss=loss, 
        train_op=train_op,
        training_hooks=training_hooks)
  
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    eval_metric_ops = {"accuracy": accuracy}
  
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss, 
      eval_metric_ops=eval_metric_ops)

  return model_fn


if __name__=="__main__":
  x = tf.placeholder(tf.float32, (None, 64, 64, 3))
  discriminator({'image':x}, 10)
  #features = {
  #  "image": tf.placeholder(tf.float32, (None, 32, 32, 3))}
  #labels = tf.placeholder(tf.int32, (None, 1))

  #model_fn_closure('test')(features, labels, tf.estimator.ModeKeys.TRAIN, {})
