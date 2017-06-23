"""Functions for Python 2 vs. 3 compatibility."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

import string
import zipfile

from BatchGenerator import BatchGenerator


class DeepFormat():
  def __init__(self):
    self.image_size = 28
    self.num_labels = 10

    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    self.train_subset = 10000

    self.num_steps = 801

    self.batch_size = 128

    self.num_hidden_nodes = 1024

    # LSTM Udacity Lesson 6
    self.url = 'http://mattmahoney.net/dc/'
    self.valid_size = 1000

    self.vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
    self.first_letter = ord(string.ascii_lowercase[0])

  # Lessons 2 & 3
  def load_mnist(self, pickle_file):
    '''

    :param pickle_file:
    :return:
    '''

    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      self.train_dataset = save['train_dataset']
      self.train_labels = save['train_labels']
      self.valid_dataset = save['valid_dataset']
      self.valid_labels = save['valid_labels']
      self.test_dataset = save['test_dataset']
      self.test_labels = save['test_labels']
      del save  # hint to help gc free up memory
      print('Training set', self.train_dataset.shape, self.train_labels.shape)
      print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
      print('Test set', self.test_dataset.shape, self.test_labels.shape)

  def reformat(self, dataset, labels):

    dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(self.num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

  def format_datasets(self):
    self.train_dataset, self.train_labels = self.reformat(self.train_dataset, self.train_labels)
    self.valid_dataset, self.valid_labels = self.reformat(self.valid_dataset, self.valid_labels)
    self.test_dataset, self.test_labels = self.reformat(self.test_dataset, self.test_labels)
    print('Training set', self.train_dataset.shape, self.train_labels.shape)
    print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
    print('Test set', self.test_dataset.shape, self.test_labels.shape)

  # Lesson 6 LSTM
  def maybe_download(self, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
      filename, _ = urlretrieve(self.url + filename, filename)
    self.statinfo = os.stat(filename)
    if self.statinfo.st_size == expected_bytes:
      print('Found and verified %s' % filename)
    else:
      print(self.statinfo.st_size)
      raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

  def read_data(self, filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
      return tf.compat.as_str(f.read(name))
    f.close()

  def format_lstm_file(self, filename):
    text = self.read_data(filename)
    print('Data size %d' % len(text))

    self.valid_text = text[:self.valid_size]
    self.train_text = text[self.valid_size:]
    self.train_size = len(self.train_text)
    print(self.train_size, self.train_text[:64])
    print(self.valid_size, self.valid_text[:64])

    self.train_batches = BatchGenerator(self.train_text)
    self.valid_batches = BatchGenerator(self.valid_text, batch_size=1, num_unrollings=1)
