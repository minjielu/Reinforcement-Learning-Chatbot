# This file has two classes.
# The Seq2Seq class builds a seq2seq model. The forward, backward maximum likelihood model
# and the reinforcement learning policy are all built by this model.
# The Critic class builds a critic model which is essentially an encoder followed by a dense layer
# which outputs an advantage.

import os
import pandas as pd
import numpy as np
import operator, csv
import tensorflow as tf
import time

class Seq2Seq():
    def build_model(self):
        x_seq_length = 20
        y_seq_length = 21
        vocabulary_size = 10004

        rnn_size = 1000
        embed_size = 1000
        num_layers = 1
        keep_prob = 1

        # Tensor where we will feed the data into graph.
        inputs = tf.placeholder(tf.float32, (None, None, 102), 'inputs')
        outputs = tf.placeholder(tf.float32, (None, None, 102), 'output')
        targets = tf.placeholder(tf.int32, (None, None), 'targets')
        masks = tf.placeholder(tf.float32, (None, None), 'weights')
        r_mask = tf.placeholder(tf.float32, (None), 'r_mask')
        RI_targets = tf.placeholder(tf.int32, (None, 2), 'flatten_targets')

        # Embedding layers are replaced by the Google Word2Vec embedding.
        # Encoding.
        encoding_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)\
                                                    for _ in range(num_layers)])

        h_input, input_states = tf.nn.dynamic_rnn(encoding_cells, inputs, dtype = tf.float32)

        # Decoding.
        decoding_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)\
                                                    for _ in range(num_layers)])

        target_sequence_length = tf.placeholder(tf.int32, [None], name = 'target_sequence_length')
        helper = tf.contrib.seq2seq.TrainingHelper(outputs, target_sequence_length)
        output_layer = tf.layers.Dense(vocabulary_size)
        decoding_layer = tf.contrib.seq2seq.BasicDecoder(decoding_cells, helper, input_states, output_layer)
        h_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoding_layer, impute_finished = True, maximum_iterations = y_seq_length)

        training_logits = tf.identity(h_output.rnn_output, name='logits')
        # Normal maximum likelihood loss function
        MLE_loss = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        # MLE Optimizer
        MLE_Optimizer = tf.train.RMSPropOptimizer(1e-4).minimize(MLE_loss)

        # Calculate the softmax possibility.
        flatten_its = tf.math.exp(tf.reshape(training_logits,[-1, training_logits.shape[-1]]))
        its_sums = tf.math.reduce_sum(flatten_its, axis = 1)
        masks_flatten = tf.reshape(masks,[-1])
        its_target = tf.gather_nd(flatten_its, RI_targets)
        target_p = tf.div(its_target, its_sums)

        # sum(A(t)*P(a(t)|s(t)))
        RI_loss = tf.tensordot(tf.log(target_p), masks_flatten, 1)
        RI_learner = tf.train.RMSPropOptimizer(1e-3)
        # Perform gradient clipping.
        gvs = RI_learner.compute_gradients(RI_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gvs]
        RI_Optimizer = RI_learner.apply_gradients(gvs)

        return inputs, outputs, targets, input_states, masks, r_mask, target_sequence_length, MLE_Optimizer, MLE_loss, training_logits, RI_Optimizer, RI_targets

class Critic():
    def __init__(self, matrix_size):
        self.matrix_size = matrix_size

    def build_model(self):

        x_seq_length = 20
        vocabulary_size = 10004

        rnn_size = 1000
        num_layers = 1
        keep_prob = 1

        states = tf.placeholder(tf.float32, (None, None, 102), 'states')
        rewards = tf.placeholder(tf.float32, (None), 'rewards')
        transform_matrix = tf.placeholder(tf.float32, (self.matrix_size, self.matrix_size), 'matrix')

        # Encoding layer of the Critic.
        encoding_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)\
                                                    for _ in range(num_layers)])

        # Encoding.
        h_input, input_states = tf.nn.dynamic_rnn(encoding_cells, states, dtype = tf.float32)

        h_input = h_input[:,-1]

        # V(s(t)) for each states
        states_V = tf.layers.dense(h_input, 1)

        # lambda*V(s(t+1))
        states_R = tf.matmul(transform_matrix, states_V)
        states_R = tf.reshape(states_R, [-1])

        # R(t) + lambda*V(s(t+1))
        discounted_rewards = tf.math.add(rewards, states_R)
        states_V = tf.reshape(states_V, [-1])

        # R(t) + lambda*V(s(t+1)) - V(s(t))
        advantages = tf.math.subtract(discounted_rewards, states_V)

        # sum of (R(t) + lambda*V(s(t+1)) - V(s(t)))^2
        critic_loss = tf.math.reduce_sum(tf.math.square(advantages))

        critic_optimizer = tf.train.RMSPropOptimizer(5e-3).minimize(critic_loss)

        return states, rewards, transform_matrix, advantages, critic_loss, critic_optimizer
