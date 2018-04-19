from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import tensorflow as tf


class CelebAInput:

  def __init__(self, crop_size=64):
    self.HEIGHT_ORIG = 218
    self.WIDTH_ORIG = 178

    self.HEIGHT = crop_size
    self.WIDTH = crop_size
    self.DEPTH = 3
    
    self.read_buffer = 8*1024*1024
    self.shuffle_buffer = 10000
    self.file_list = ['data/img_align_celeba.tfrecords']

  def make_input_fn(self, mode='test', batch_size=64):
    def input_fn():
      file_list = self.file_list
      if type(file_list) is list and len(file_list)>1:
        file_list = self.file_list
        dataset = tf.data.Dataset.list_files(file_list)
      
        if mode == 'train':
          dataset = dataset.shuffle(buffer_size=min(len(file_list), 1024)).repeat()
      
        def process_tfrecord(file_name):
          dataset = tf.data.TFRecordDataset(file_name, buffer_size=self.read_buffer)
          return dataset

        dataset = dataset.apply(
          tf.contrib.data.parallel_interleave(
            process_tfrecord, cycle_length=4, sloppy=True))
      else:
        dataset = tf.data.TFRecordDataset(file_list, buffer_size=self.read_buffer)
      
      if mode=='train':
        dataset = dataset.shuffle(self.shuffle_buffer)

      dataset = dataset.map(self.make_parser(mode), num_parallel_calls=64)
      dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

      dataset = dataset.prefetch(batch_size)
    
      # TODO:
      # - reinitializable iterator
      # - add noise
      image = dataset.make_one_shot_iterator().get_next()
      return image

    return input_fn
    
  def make_parser(self, mode):
    def parser_fn(value):
      keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string, '')}
      parsed = tf.parse_single_example(value, features=keys_to_features)
      image = tf.decode_raw(parsed['image'], tf.uint8)
      image.set_shape([self.HEIGHT_ORIG * self.WIDTH_ORIG * self.DEPTH])
      
      image = tf.cast(image, tf.float32)
     
      image = tf.transpose(
        tf.reshape(2*image/255.0 - 1, (self.DEPTH, self.HEIGHT_ORIG, self.WIDTH_ORIG)), 
        (1,2,0))
      
      image = tf.image.resize_images(
                image, 
                (self.HEIGHT, self.WIDTH), 
                method=tf.image.ResizeMethod.BICUBIC)

      return image
    return parser_fn


if __name__=='__main__':
  celeba = CelebAInput()
  input_fn = celeba.make_input_fn(mode='train', batch_size=2)

  with tf.Session() as sess:
    image = input_fn()
    
    for i in range(10):
      images = sess.run(image)
      print(images.shape)
