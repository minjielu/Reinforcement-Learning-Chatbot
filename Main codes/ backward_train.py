# This file trains a backward maximum likelihood model.

import os
import pandas as pd
import numpy as np
import operator, csv
import tensorflow as tf
import time
from build_model import Seq2Seq
from batch_generator import batch_generator, word2vec_lookup

start_new = False
x_seq_length = 20
y_seq_length = 21
# Load the vocabulary.
vocabulary = np.loadtxt('Vocabulary.csv', dtype = object, delimiter = '###')
wordDict = dict(zip(vocabulary[1:], range(len(vocabulary) - 1)))
wordDict['<UNK>'] = len(vocabulary) - 1
wordDict['<PAD>'] = len(vocabulary)
wordDict['<START>'] = len(vocabulary) + 1
wordDict['<END>'] = len(vocabulary) + 2
wordDict['<NAME>'] = len(vocabulary) + 3

# Hyperparameters.
epochs = 2
batch_size = 128
rnn_size = 1000
embed_size = 100
num_layers = 1
keep_prob = 1

# Build the model.
inputs, outputs, targets, input_states, masks, r_mask, \
target_sequence_length, MLE_optimizer, MLE_loss, training_logits, \
RI_loss, RI_optimizer, flatten_targets = Seq2Seq().build_model()

# Build the word2vec embedding.
w2v = word2vec_lookup('word2vec_dict.model')

with tf.Session() as sess:
    # Start to train a new model or continue to train an old model.
    if start_new:
        sess.run(tf.global_variables_initializer())
    else:
        saver = tf.train.Saver()
        saver.restore(sess, "./models/backward_model")
    epochs = 2
    loss_acc_log = []

    for epoch_i in np.arange(epochs):
        # Get the next batch of data.
        start_time = time.time()
        bg1 = batch_generator('Outputs/', batch_size)
        bg2 = batch_generator('Inputs/', batch_size)
        over_1, source_batch = bg1.next_batch()
        over_2, answer_batch = bg2.next_batch()
        batch_num = 0

        while over_1 and over_2:
            batch_num += 1

            # Pad all sources and targets.
            source_maxlen = max([len(x) for x in source_batch])
            source_batch = [[wordDict['<PAD>']]*(source_maxlen-len(x))+x for x in source_batch]
            target_batch = [x+[wordDict['<END>']] for x in answer_batch]
            answer_batch = [[wordDict['<START>']]+x for x in answer_batch]
            target_maxlen = max([len(x) for x in target_batch])
            mask_batch = [[1]*len(x)+[0]*(target_maxlen-len(x)) for x in target_batch]
            target_batch = np.asarray([x+[wordDict['<PAD>']]*(target_maxlen-len(x)) for x in target_batch])
            answer_batch = [x+[wordDict['<PAD>']]*(target_maxlen-len(x)) for x in answer_batch]

            target_lengths = np.repeat(target_maxlen, batch_size)

            # Embed source_batch and answer_batch using the word2vec embedding.
            source_batch = [[w2v.lookup(x) for x in y] for y in source_batch]
            answer_batch = [[w2v.lookup(x) for x in y] for y in answer_batch]

            # Perform training.
            _, batch_loss, batch_logits = sess.run([MLE_optimizer, MLE_loss, training_logits],
                feed_dict = {inputs: source_batch,
                 outputs: answer_batch,
                 targets: target_batch,
                 masks: mask_batch,
                 target_sequence_length: target_lengths})
            if batch_num % 10 == 0:
                print('{} batches trained:'.format(batch_num))
                accuracy = np.mean((batch_logits.argmax(axis=-1) == target_batch)[np.nonzero(mask_batch)])
                print('Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(batch_loss, \
                                                                                accuracy, time.time() - start_time))

            # Save the model and training performance.
            if batch_num % 100 == 0:
                loss_acc_log += [batch_loss, accuracy]
                saver = tf.train.Saver()
                saver.save(sess, './models/backward_model_1')
                np.savetxt('./logs/training_history_1.csv', loss_acc_log, delimiter = ',')
                print('Model and log are saved!')
            over_1, source_batch = bg1.next_batch()
            over_2, answer_batch = bg2.next_batch()
