import struct
import os
import urllib
import zipfile
import pickle
import PIL
import numpy as np
import tensorflow as tf


def jpg_reader(filename):
  img = PIL.Image.open(filename)
  return np.asarray(img)


def to_tfrecords():
  img_dir = 'data/img_align_celeba'
  
  if not os.path.isdir('data'):
    os.mkdir('data')
  
  assert os.path.isdir(img_dir), 'Download the aligned image celeba .zip and unzip into the data/ directory'
  
  input_files = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir) if fn.endswith(".jpg")]
  output_file = 'data/img_align_celeba.tfrecords'

  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      img = jpg_reader(input_file)

      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))}))

      record_writer.write(example.SerializeToString())


if __name__ == '__main__':
  to_tfrecords()
