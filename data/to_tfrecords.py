import struct
import os
import pickle
import PIL
import numpy as np
import tensorflow as tf


def jpg_reader(filename):
  img = PIL.Image.open(filename)
  return np.asarray(img)


if __name__=="__main__":
  
  img_dir = 'img_align_celeba'
  assert os.path.isdir(img_dir), 'Download the aligned image celeba zip'
  
  input_files = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir) if fn.endswith(".jpg")]
  output_file = 'img_align_celeba.tfrecords'

  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      img = jpg_reader(input_file)

      example = tf.train.Example(
        features=tf.train.Features(
          feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))}))

      record_writer.write(example.SerializeToString())
