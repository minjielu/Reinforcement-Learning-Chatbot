import os
import numpy as np
import tensorflow as tf

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
        flatten_targets = tf.placeholder(tf.int32, (None, 2), 'flatten_targets')

        # Embedding layers are replaced by the Google Word2Vec embedding.

        encoding_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)\
                                                    for _ in range(num_layers)])

        h_input, input_states = tf.nn.dynamic_rnn(encoding_cells, inputs, dtype = tf.float32)

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
        MLE_optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(MLE_loss)
        # Reinforcement learning loss function
        flatten_logits = tf.reshape(training_logits, [-1, vocabulary_size])
        entropy = tf.gather_nd(flatten_logits, flatten_targets)
        RI_loss = tf.tensordot(entropy, r_mask, axes = 1)
        # Reinforcement learning optimizer
        RI_optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(RI_loss)
        return inputs, outputs, targets, input_states, masks, r_mask, target_sequence_length, MLE_optimizer, MLE_loss, training_logits, RI_loss, RI_optimizer, flatten_targets
