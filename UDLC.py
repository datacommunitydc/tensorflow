from __future__ import print_function
import numpy as np
import tensorflow as tf

from six.moves import range

# From Udacity lesson 6 LSTM
import os
import random

from DeepFormat import DeepFormat
# from DeepUtilities import DeepUtilities

class UDLC(DeepFormat):

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
    self.valid_size = 1000

  def create_graph(self,l2_regularization=False):

    self.graph = tf.Graph()
    with self.graph.as_default():

      if l2_regularization:
        # Input data.
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        self.tf_train_dataset = tf.placeholder(tf.float32,
                                               shape=(self.batch_size, self.image_size**2))
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
        self.tf_valid_dataset = tf.constant(self.valid_dataset)
        self.tf_test_dataset = tf.constant(self.test_dataset)
        self.beta = tf.placeholder(tf.float32)

        # Variables.
        self.weights = tf.Variable(
          tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
        self.biases = tf.Variable(tf.zeros([self.num_labels]))

        # Training computation.
        self.logits = tf.matmul(self.tf_train_dataset, self.weights) + self.biases
        self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

        self.regularization = self.beta * tf.nn.l2_loss(self.weights)
        self.loss += self.regularization

        tf.scalar_summary('loss', self.loss)

        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        self.train_prediction = tf.nn.softmax(self.logits)
        self.valid_prediction = tf.nn.softmax(
          tf.matmul(self.tf_valid_dataset, self.weights) + self.biases)
        self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, self.weights) + self.biases)

      else:
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        self.tf_train_dataset = tf.constant(self.train_dataset[:self.train_subset, :])
        self.tf_train_labels = tf.constant( self.train_labels[:self.train_subset])
        self.tf_valid_dataset = tf.constant(self.valid_dataset)
        self.tf_test_dataset = tf.constant( self.test_dataset)

        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        self.weights = tf.Variable(
          tf.truncated_normal([self.image_size**2, self.num_labels]))
        self.biases = tf.Variable(tf.zeros([self.num_labels]))

        print('weights',self.weights)
        print('biases',self.biases)

        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        self.logits = tf.matmul(self.tf_train_dataset, self.weights) + self.biases
        self.loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(self.logits, self.tf_train_labels))

        # Optimizer.
        self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)


        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        self.train_prediction = tf.nn.softmax(self.logits)
        self.valid_prediction = tf.nn.softmax(
          tf.matmul(self.tf_valid_dataset, self.weights) + self.biases)
        self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, self.weights) + self.biases)

  def accuracy(self, predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

  def train_graph(self,batch_processing=False):

    with tf.Session(graph=self.graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases.
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(self.num_steps):

        if batch_processing:
          # Pick an offset within the training data, which has been randomized.
          # Note: we could use better randomization across epochs.
          offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
          # Generate a minibatch.
          batch_data = self.train_dataset[offset:(offset + self.batch_size), :]
          batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
          # Prepare a dictionary telling the session where to feed the minibatch.
          # The key of the dictionary is the placeholder node of the graph to be fed,
          # and the value is the numpy array to feed to it.
          feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
          _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)

        else:
          # Run the computations. We tell .run() that we want to run the optimizer,
          # and get the loss value and the training predictions returned as numpy
          # arrays.
          _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction])


        if (step % 100 == 0):
          print('Loss at step %d: %f' % (step, l))
          print('Training accuracy: %.1f%%' % self.accuracy(
            predictions, self.train_labels[:self.train_subset, :]))
          # Calling .eval() on valid_prediction is basically like calling run(), but
          # just to get that one numpy array. Note that it recomputes all its graph
          # dependencies.
          print('Validation accuracy: %.1f%%' % self.accuracy(
            self.valid_prediction.eval(), self.valid_labels))
        print('Test accuracy: %.1f%%' % self.accuracy(self.test_prediction.eval(), self.test_labels))

