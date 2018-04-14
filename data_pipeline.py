from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import tensorflow as tf


class CelebAInput:

  def __init__(self):
    self.HEIGHT = 218
    self.WIDTH = 178
    self.DEPTH = 3
    
    self.read_buffer = 8*1024*1024
    self.shuffle_buffer = 10000
    self.file_list = ['data/img_align_celeba.tfrecords']

  def input_fn_factory(self, mode='test', batch_size=64):
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

      dataset = dataset.map(self.parser_factory(mode), num_parallel_calls=64)
      dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

      dataset = dataset.prefetch(batch_size)
    
      image = dataset.make_one_shot_iterator().get_next()
      return {'image': image} 

    return input_fn
    
  def parser_factory(self, mode):
    def parser_fn(value):
      keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string, '')}
      parsed = tf.parse_single_example(value, features=keys_to_features)
      image = tf.decode_raw(parsed['image'], tf.uint8)
      image.set_shape([self.HEIGHT * self.WIDTH * self.DEPTH])
      
      image = tf.cast(
        tf.transpose(tf.reshape(image, [self.DEPTH, self.HEIGHT, self.WIDTH]), [1,2,0]),
        tf.float32)
      
      if mode == 'train':
        image = self.preprocess_fn(image)
      
      image = tf.image.per_image_standardization(image)
      return image

    return parser_fn

  def preprocess_fn(self, image):
    image = tf.image.resize_image_with_crop_or_pad(image, self.HEIGHT+16, self.WIDTH+16)
    image = tf.random_crop(image, [self.HEIGHT, self.WIDTH, self.DEPTH])
    image = tf.image.random_flip_left_right(image)
    return image


if __name__=='__main__':
  celeba = CelebAInput()
  input_fn = celeba.input_fn_factory(mode='train', batch_size=2)

  with tf.Session() as sess:
    image = input_fn()
    
    for i in range(10):
      images = sess.run(image['image'])
      print(images.shape)

