import sys
import numpy as np
import tensorflow as tf


if __name__=='__main__':
  new_size = int(sys.argv[1])

  a = np.asarray([
    [1, 0, 0, 2],
    [2, 4, 0, 1],
    [3, 0, 2, 0],
    [4, 3, 2, 1]]).astype(np.uint8)[:,:,None]

  x = tf.placeholder(tf.uint8, (4,4,1))
  y = tf.image.resize_images(x, (new_size, new_size), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
  with tf.Session() as sess:
    res = sess.run(y, feed_dict={x: a})
    print(a[:,:,0])
    print('-'*40)
    print(res[:,:,0])
