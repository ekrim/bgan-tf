from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class ChkptSwap:
  '''Build this object with a list of desired variables
  and a name. Call .step() for each training iteration
  with a given probability, the current model will be saved 
  in a checkpoint and will automatically be swapped in for 
  the model in the future after a randomly chosen number
  of iterations. From there, another randomly chosen number
  of iterations passes, and the swapped out model will be 
  swapped back in.
  '''
  def __init__(self, var_list, name, pr_swap=0.05):
    self.internal_cnt = 0 
    self.state = 0
    self.pr_swap = pr_swap

    self.saver = tf.train.Saver(var_list)
    self.sess = tf.get_default_session()

    self.dirs = {}
    for s in ['old', 'curr']:
      base = 'chkpts_{:s}_{:s}'.format(s, name)
      self._clear(base)
      self.dirs.update({s: os.path.join(base, 'model-chkpt')})
     
  def step(self):
    if self.state == 0 and np.random.rand() < self.pr_swap:
      self._save_current()

    elif self.state == 1 and self.internal_cnt == self.dur:
      self._swap_to_old()
    
    elif self.state == 2 and self.internal_cnt == self.dur:
      self._swap_back()
    self.internal_cnt += 1

  def _save_current(self):
    self.state = 1
    self.saver.save(self.sess, self.dirs['old'])
    self.dur = np.random.randint(1, 7)

  def _swap_to_old(self):
    self.saver.save(self.sess, self.dirs['curr'])
    self.saver.restore(self.sess, self.dirs['old'])
    self.state = 2
    self.internal_cnt = 0
    self.dur = np.random.randint(1, 5) 

  def _swap_back(self):
    self.saver.restore(self.sess, self.dirs['curr'])
    self.state = 0
    self.internal_cnt = 0

  def _clear(self, dir_name):
    if os.path.exists(dir_name):
      shutil.rmtree(dir_name)
    os.mkdir(dir_name)


class ImageBuffer:
  '''Generated images are added to this buffer
  The discriminator trains on a mix of freshly erated images
  and images from this buffer
  '''
  def __init__(self, img_dim, buffer_sz):
    self.buffer_sz = buffer_sz
    self.buffer = np.zeros((buffer_sz,)+img_dim, dtype=np.float32)
    self.cnt = 0 
  
  def add(self, x):
    if self.cnt < self.buffer_sz:
      n_to_add = min(self.buffer_sz-self.cnt, x.shape[0])
      self.buffer[self.cnt:self.cnt+n_to_add] = x[:n_to_add]
      self.cnt += n_to_add
    else:
      idx = np.random.choice(self.buffer_sz, x.shape[0], replace=False)
      self.buffer[idx] = x

  def sample(self, batch_size):
    idx = np.random.choice(self.cnt, batch_size, replace=True)
    return self.buffer[idx]


def plot_images(images, filename):
  '''plot function derived from:

  github.com/wiseodd

  thanks!
  '''
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)
  
  images = (255*(images + 1)/2).astype(np.uint8)
  for i, sample in enumerate(images):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample)
  plt.savefig(filename, bbox_inches='tight')
  plt.close(fig)


if __name__=='__main__':
  a = ImageBuffer((3,), 8)
  print('initial buffer')
  print(a.buffer)
  for i in range(3):
    a.add((i+1)*np.ones((3,3)))
    print('adding to buffer...')
    print(a.buffer)
  print('drawing 4 samples from buffer')
  print(a.sample(4))
