from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
'''
To classify text using a BiDirectional LSTM
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
# n_input = 28 # MNIST data input (img shape: 28*28)
# n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
# n_classes = 10 # MNIST total classes (0-9 digits)


class BiRNNClassifier:

    def __init__(self, sequence_length, num_classes):
        self.n_steps = sequence_length
        self.n_classes = num_classes
        # Define weights
        self.weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * n_hidden, self.n_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    def build(self):
        self.add_placeholders()
        self.add_pred_op()
        self.add_cost_op()
        self.add_optimize_op()
        self.add_accuracy_op()

    def add_placeholders(self):
        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_steps, 1])
        self.y = tf.placeholder("float", [None, self.n_classes])

    def add_pred_op(self):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(self.x, self.n_steps, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        self.pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']


    def add_cost_op(self):
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))


    def add_optimize_op(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def add_accuracy_op(self):
        correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    def train(self, dataset):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_x, batch_y = dataset.train.next_batch(batch_size)
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((len(batch_x), self.n_steps, 1))
                # Run optimization op
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                if step % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(self.accuracy, feed_dict={self.x: batch_x, self.y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(self.cost, feed_dict={self.x: batch_x, self.y: batch_y})
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")

            # Calculate accuracy for test data
            test_x, test_y = dataset.dev.all()
            test_x = test_x.reshape((len(test_x), self.n_steps, 1))
            print("Testing Accuracy:", sess.run(self.accuracy, feed_dict={self.x: test_x, self.y: test_y}))