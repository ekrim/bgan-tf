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
    
    self.read_buffer = 3*64*64*2**13
    self.shuffle_buffer = 2**15
    self.file_list = ['data/img_align_celeba.tfrecords']

  def input_fn(self, mode='test', epochs=1, batch_size=64):
    file_list = self.file_list
    if type(file_list) is list and len(file_list)>1:
      print('\n***processing in multiple file interleave mode***\n')
      file_list = self.file_list
      dataset = tf.data.Dataset.list_files(file_list, shuffle=True)
    
      if mode == 'train':
        dataset = dataset.shuffle(
	  buffer_size=min(len(file_list), self.shuffle_buffer)).repeat()
    
      def process_tfrecord(file_name):
        dataset = tf.data.TFRecordDataset(
	  file_name, 
	  buffer_size=self.read_buffer)
        return dataset

      dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
          process_tfrecord, cycle_length=4, sloppy=True))
    else:
      print('\n*** processing in single file mode ***\n')
      dataset = tf.data.TFRecordDataset(
        file_list, 
	buffer_size=self.read_buffer)
    
    if mode=='train':
      dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
        self.shuffle_buffer, 
	count=epochs))

    dataset = dataset.map(
      self.make_parser(mode), 
      num_parallel_calls=8)

    dataset = dataset.batch(batch_size)
    #dataset = dataset.apply(
    #  tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(batch_size)
   
    iterator = dataset.make_initializable_iterator()
    image = iterator.get_next()
    
    return image, iterator
    
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

  image, iterator = CelebAInput().input_fn(
    mode='train', 
    batch_size=2048, 
    epochs=1)

  with tf.Session() as sess:
    sess.run(iterator.initializer)
    cnt = 0
    while True:
      try:
        images = sess.run(image)
        print(images.shape)
        cnt += images.shape[0]
        print(cnt)
      except tf.errors.OutOfRangeError:
        break
