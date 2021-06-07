import numpy as np
import tensorflow as tf
import random
import tf_slim as slim


class Qnetwork():
    def __init__(self, h_size, scope):
        with tf.compat.v1.variable_scope(scope):
            # The network recieves a frame from the game, flattened into an array.
            # It then resizes it and processes it through convolutional layers.
            self.imageIn = tf.compat.v1.placeholder(shape=[None, 15, 15, 4], dtype=tf.float32)
            self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=16, kernel_size=[3, 3], stride=[1, 1],
                                     padding='VALID', biases_initializer=None)
            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32, kernel_size=[3, 3], stride=[1, 1],
                                     padding='VALID', biases_initializer=None)
            self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=32, kernel_size=[3, 3], stride=[1, 1],
                                     padding='VALID', biases_initializer=None)

            # We take the output from the final convolutional layer and split it into separate advantage and value streams.
            self.streamAC, self.streamVC = tf.split(self.conv3, 2, 3)
            self.streamA = slim.flatten(self.streamAC)
            self.streamV = slim.flatten(self.streamVC)
            xavier_init = tf.keras.initializers.GlorotNormal()
            self.AW = tf.Variable(xavier_init([h_size // 2, 4]))
            self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
            self.predict = tf.argmax(self.Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)


# Experience buffer for DQN
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


# Function for getting training value target network
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))

    return op_holder


# Function for updating target network
def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)