from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

class UDLC():

    def __init__(self):
        self.image_size = 28
        self.num_labels = 10

        # With gradient descent training, even this much data is prohibitive.
        # Subset the training data for faster turnaround.
        self.train_subset = 10000

    def load_mnist(self,pickle_file):
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

    def reformat(self,dataset,labels):

      self.dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
      # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
      self.labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
      # return dataset, labels

    def format_datasets(self):
        self.train_dataset, train_labels = self.reformat(self.train_dataset, self.train_labels)
        self.valid_dataset, valid_labels = self.reformat(self.valid_dataset, self.valid_labels)
        self.test_dataset, test_labels = self.reformat(self.test_dataset, self.test_labels)
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)

    def create_graph(self):

        graph = tf.Graph()
        with graph.as_default():

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
            weights = tf.Variable(
            tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
            biases = tf.Variable(tf.zeros([self.num_labels]))

            print('weights',weights)
            print('biases',biases)

            # Training computation.
            # We multiply the inputs with the weight matrix, and add biases. We compute
            # the softmax and cross-entropy (it's one operation in TensorFlow, because
            # it's very common, and it can be optimized). We take the average of this
            # cross-entropy across all training examples: that's our loss.
            logits = tf.matmul(tf_train_dataset, weights) + biases
            loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
            tf.scalar_summary('loss', loss)

            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
            test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)



        num_steps = 801

