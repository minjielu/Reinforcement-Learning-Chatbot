# This file performs the advantage actor critic algorithm.

import os
import pandas as pd
import numpy as np
import operator, csv
import tensorflow as tf
import time
import heapq as hq
from BuildModel import Seq2Seq, Critic
from batch_generator import batch_generator, word2vec_lookup

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
w2v = word2vec_lookup('word2vec_dict.model')

# Hyperparameters.
epochs = 2
batch_size = 128
rnn_size = 1000
embed_size = 100
num_layers = 1
keep_prob = 1
max_num_infers = 6
max_turns = 1
reward_lambda = 0.6
num_actors = 32
train_only_critic = True # Whether only the pretrain of the critic is performed.
alphaGo_manner = False # Whether a single agent is used or two agents talk with each other.
rewards_log = []

# Construct the matrix which is useful for calculating the advantages.
# Basically matrix for transfering r(1), r(2), ... and V(1), V(2), V(3)...
# to R(1) = r(1)+d*r(2)+d^2*r(3)...
matrix_size = (max_turns+1)*num_actors
reward_transform_matrix = np.zeros([matrix_size, matrix_size])
for k in np.arange(num_actors):
    for l in np.arange(max_turns):
        for m in np.arange(l, max_turns):
            reward_transform_matrix[k*(max_turns+1)+m,k*(max_turns+1)+l] = reward_lambda**(m-l)
V_transform_matrix = np.zeros([matrix_size, matrix_size])
for k in np.arange(max_turns, matrix_size, max_turns+1):
    for l in np.arange(k,k-max_turns-1,-1):
        V_transform_matrix[l,k] = reward_lambda**(k-l)


tf.reset_default_graph()
# Load the policy model.
sess_1 = []
g_1_1 = tf.Graph()
inputs_1 = []
outputs_1 = []
targets_1 = []
input_states_1 = []
masks_1 = []
r_mask_1 = []
target_sequence_length_1 = []
MLE_optimizer_1 = []
loss_1 = []
training_logits_1 = []
RI_optimizer_1 = []
RI_targets_1 = []
critic_loss_log = []

# Load the policy models.
# Load the first agent.
with g_1_1.as_default():
    graph_1_1 = Seq2Seq()
    inputs_1_1, outputs_1_1, targets_1_1, input_states_1_1, masks_1_1, r_mask_1_1, \
    target_sequence_length_1_1, MLE_optimizer_1_1, loss_1_1, \
    training_logits_1_1, RI_optimizer_1_1, RI_targets_1_1 = graph_1_1.build_model()
    inputs_1.append(inputs_1_1)
    outputs_1.append(outputs_1_1)
    targets_1.append(targets_1_1)
    input_states_1.append(input_states_1_1)
    masks_1.append(masks_1_1)
    r_mask_1.append(r_mask_1_1)
    target_sequence_length_1.append(target_sequence_length_1_1)
    MLE_optimizer_1.append(MLE_optimizer_1_1)
    loss_1.append(loss_1_1)
    training_logits_1.append(training_logits_1_1)
    RI_optimizer_1.append(RI_optimizer_1_1)
    RI_targets_1.append(RI_targets_1_1)
    saver_1_1 = tf.train.Saver()
    sess_1.append(tf.Session(graph = g_1_1))
    saver_1_1.restore(sess_1[0], "./Models/forward_model_1")

# Load the second agent.
g_1_2 = tf.Graph()
with g_1_2.as_default():
    graph_1_2 = Seq2Seq()
    inputs_1_2, outputs_1_2, targets_1_2, input_states_1_2, masks_1_2, r_mask_1_2, \
    target_sequence_length_1_2, MLE_optimizer_1_2, loss_1_2, \
    training_logits_1_2, RI_optimizer_1_2, RI_targets_1_2 = graph_1_2.build_model()
    saver_1_2 = tf.train.Saver()
    inputs_1.append(inputs_1_2)
    outputs_1.append(outputs_1_2)
    targets_1.append(targets_1_2)
    input_states_1.append(input_states_1_2)
    masks_1.append(masks_1_2)
    r_mask_1.append(r_mask_1_2)
    target_sequence_length_1.append(target_sequence_length_1_2)
    MLE_optimizer_1.append(MLE_optimizer_1_2)
    loss_1.append(loss_1_2)
    training_logits_1.append(training_logits_1_2)
    RI_optimizer_1.append(RI_optimizer_1_2)
    RI_targets_1.append(RI_targets_1_2)
    sess_1.append(tf.Session(graph = g_1_2))
    saver_1_2.restore(sess_1[1], "./Models/forward_model_1")

tf.reset_default_graph()
# Load the forward Seq2Seq model.
g_2 = tf.Graph()
with g_2.as_default():
    graph_2 = Seq2Seq()
    inputs_2, outputs_2, targets_2, input_states_2, masks_2, r_mask_2, \
    target_sequence_length_2, MLE_optimizer_2, loss_2, \
    training_logits_2, RI_Optimizer_2, RI_targets_2 = graph_2.build_model()
    saver_2 = tf.train.Saver()
    sess_2 = tf.Session(graph = g_2)
    saver_2.restore(sess_2, "./Models/forward_model")

tf.reset_default_graph()
# Load the backward Seq2Seq model.
g_3 = tf.Graph()
with g_3.as_default():
    graph_3 = Seq2Seq()
    inputs_3, outputs_3, targets_3, input_states_3, masks_3, r_mask_3, \
    target_sequence_length_3, MLE_optimizer_3, loss_3, \
    training_logits_3, RI_Optimizer_3, RI_targets_3 = graph_3.build_model()
    saver_3 = tf.train.Saver()
    sess_3 = tf.Session(graph = g_3)
    saver_3.restore(sess_3, "./Models/backward_model")

tf.reset_default_graph()
# Load the critic.
g_4 = tf.Graph()
with g_4.as_default():
    graph_4 = Critic(matrix_size)
    states_for_critic, rewards_for_critic, transform_matrix_for_critic, \
    advantages_from_critic, loss_of_critic, critic_optimizer = graph_4.build_model()
    saver_4 = tf.train.Saver()
    sess_4 = tf.Session(graph = g_4)
    # sess_4.run(tf.global_variables_initializer())
    saver_4.restore(sess_4, "./Models/Critic")

# Generate a dialogue starting from each utterances in the dataset and calculate the rewards

bg1 = batch_generator('Inputs/', 1)

over_1, p1 = bg1.next_batch()

#-------------------------------------------------------------------------------
cnt = 0
while over_1:
    cnt += 1
    inputs = []
    answers = []
    answer_targets = []
    states = []
    rewards = []
    last_p1 = []
    turn = 0
    batch_reward = 0
    critic_grad = []

    for k in np.arange(0, num_actors):
        # q1 = q1[0]
        p1 = p1[0]
        # Actor take actions----------------------------------------------------
        for l in np.arange(0,max_turns):
            inputs.append(p1) #inputs.append(p1+q1)
            states.append(p1)
            input = [p1] #input = [p1 + q1]
            # print(input)
            output = [[wordDict['<START>']]]
            input_embeded = [[w2v.lookup(x) for x in y] for y in input]
            heap_queue = []
            infers = []
            infer_pros = []
            hq.heappush(heap_queue, (0, [wordDict['<START>']]))
            # Possible actions are searched using a single source shortest path manner.
            while 1:
                cur_infer = hq.heappop(heap_queue)
                cur_score = cur_infer[0]
                output = [cur_infer[1]]
                if output[0][-1] == wordDict['<END>'] or len(output[0]) == y_seq_length:
                    # The actions without the <START> and <END> sign are pushed in.
                    infers.append(output[0])
                    infer_pros.append(cur_score)
                    if len(infers) == max_num_infers:
                        break
                    continue
                output_embeded = [[w2v.lookup(x) for x in y] for y in output]
                logits = sess_1[turn].run(training_logits_1[turn],
                         feed_dict = {inputs_1[turn]: input_embeded,
                         outputs_1[turn]: output_embeded,
                         target_sequence_length_1[turn]: np.asarray([len(output[0])])})
                indexes = logits[0,-1].argsort(axis = -1)[-3:].reshape(-1)
                logits = np.exp(logits[0,-1][indexes])
                logits = -np.log(logits/np.sum(logits))
                for n in np.arange(3):
                    hq.heappush(heap_queue, (cur_score+logits[n], output[0]+[indexes[n]]))

            choice = np.random.randint(0, len(infers))
            output = [infers[choice]]

            dull_set = ["i don' t know what you' re talking about .", "i don' t know .", "you don' t know ."\
                        , "you know what i mean .", "i know what you mean .", "you know what i' m saying . "\
                        , "you don' know anything ."]

            action = np.asarray(output)
            action_target = action[:,1:]
            action = action[:,:-1]
            answers.append(action[0].tolist())
            action_as_input = action[:,1:]
            answer_targets.append(action_target[0].tolist())

            # Calculate rewards-------------------------------------------------

            # Easy of answering------------------

            r1 = 0
            action_as_input_embeded = [[w2v.lookup(x) for x in y] for y in action_as_input]
            '''
            for dull_answer in dull_set:
                dull_answer = np.asarray([[wordDict['<START>']] + [wordDict[x] if x in wordDict else wordDict['<UNK>'] for x in dull_answer.split()]])
                dull_answer_embeded = [[w2v.lookup(x) for x in y] for y in dull_answer]
                logits = sess_2.run(training_logits_2,
                         feed_dict = {inputs_2: action_as_input_embeded,
                                      outputs_2: dull_answer_embeded,
                                      target_sequence_length_2: np.asarray([len(dull_answer[0])])})
                its = np.exp(logits[0])
                sums = np.sum(its, axis = 1)
                dull_answer = np.concatenate((dull_answer[:,1:], np.full((dull_answer.shape[0],1), wordDict['<END>'])), axis = 1)
                r1 += np.sum(np.log([its[i,j]/sums[i] for i, j in enumerate(dull_answer[0],0)]))/len(dull_answer[0])
            r1 /= -len(dull_set)
            #r1 = np.exp(r1)

            r2 = 0

            if l > 0:
                # Information flow--------------
                last_p1_embeded = [[w2v.lookup(x) for x in y] for y in [last_p1]]
                h_last_p1 = sess_2.run(input_states_2, feed_dict = {inputs_2: last_p1_embeded})[0]
                h_last_p1 = h_last_p1[1].flatten()
                h_p2 = sess_2.run(input_states_2, feed_dict = {inputs_2: action_as_input_embeded})[0]
                h_p2 = h_p2[1].flatten()
                r2 = -np.log(np.cos(np.dot(h_last_p1, h_p2)/np.linalg.norm(h_last_p1)/np.linalg.norm(h_p2)))
                #r2 = (1-np.dot(h_p1, h_p2)/np.linalg.norm(h_p1)/np.linalg.norm(h_p2)) / 2
            '''
            # Semantic Coherence----------------
            # Forward possibility.
            action_embeded = [[w2v.lookup(x) for x in y] for y in action]
            forward_logits = sess_2.run(training_logits_2,
                                feed_dict = {inputs_2: input_embeded,
                                            outputs_2: action_embeded,
                                            target_sequence_length_2: np.asarray([len(action[0])])})
            forward_its = np.exp(forward_logits[0])
            forward_sums = np.sum(forward_its, axis = 1)
            forward_logpro = np.sum(np.log([forward_its[i,j]/forward_sums[i] for i, j in enumerate(action_target[0],0)]))
            forward_logpro /= len(action_target[0])

            # Backward possibility.
            p1_embeded = [[w2v.lookup(x) for x in y] for y in [[wordDict['<START>']]+p1]]
            backward_logits = sess_3.run(training_logits_3,
                                feed_dict = {inputs_3: action_as_input_embeded,
                                            outputs_3: p1_embeded, #q1_embeded,
                                            target_sequence_length_3: np.asarray([len(p1)+1])})
            backward_its = np.exp(backward_logits[0])
            backward_sums = np.sum(backward_its, axis = 1)
            backward_logpro = np.sum(np.log([backward_its[i,j]/backward_sums[i] for i, j in enumerate(p1+[wordDict['<END>']],0)]))
            backward_logpro /= len(p1)+1

            r3 = forward_logpro + backward_logpro
            #r3 = np.exp(r3)
            rewards.append(r3)
            p1 = action_as_input[0].tolist()

            # Note this line if we only want a robot to chat with himself.
            # turn = 1-turn

        batch_reward += np.sum(rewards) # Raw reward for logging.
        states.append(answers[-1])
        rewards.append(0)
        over_1, p1 = bg1.next_batch()

    # Critic provides baselines---------------------------------------------
    # Padding states
    rewards = np.matmul(rewards, reward_transform_matrix)
    states_maxlen = max([len(x) for x in states])
    states = [[wordDict['<PAD>']]*(states_maxlen-len(x)) + x for x in states]
    states_embeded = [[w2v.lookup(x) for x in y] for y in states]
    advantages, critic_loss, _ = sess_4.run([advantages_from_critic, loss_of_critic, critic_optimizer],
                                    feed_dict = {
                                    states_for_critic: states_embeded,
                                    rewards_for_critic: rewards,
                                    transform_matrix_for_critic: V_transform_matrix,
                                    })

    # Save the critic model and its performance.
    if cnt % 10 == 0:
        print('Loss of critic for batch_{}: {}'.format(cnt, critic_loss))
        critic_loss_log.append(critic_loss)
    if cnt % 200 == 0:
        saver_4.save(sess_4, './models/Critic_1')
        np.savetxt('./logs/Critic_history_1.csv', critic_loss_log, delimiter = ',')
        print('Critic model and log are saved!')

    # Train the policy---------------
    if not train_only_critic:

        rewards = advantages[np.arange(matrix_size) % (max_turns+1) != max_turns]

        if alphaGo_manner:
            # If an alphaGo manner is used. Basically, two agents talk with each other.
            all_inputs = inputs
            all_answers = answers
            all_answer_targets = answer_targets
            all_rewards = rewards

            for k in np.arange(2):
                inputs = []
                answers = []
                answer_targets = []
                rewards = []
                for l in np.arange(k,max_turns,2):
                    inputs.append(all_inputs[l])
                    answers.append(all_answers[l])
                    answer_targets.append(all_answer_targets[l])
                    rewards.append(all_rewards[l])

                # Pad the states.
                inputs_maxlen = max([len(x) for x in inputs])
                inputs = [[wordDict['<PAD>']]*(inputs_maxlen-len(x)) + x for x in inputs]
                r_mask = []
                # Train the network.
                # Generate a list of reward for each action.
                answer_maxlen = max([len(answer) for answer in answers])
                for i, answer in enumerate(answers):
                    r_mask.append([rewards[i]*(-1)]*len(answer) + [0]*(answer_maxlen - len(answer)))

                # Pad the actions.
                answers = np.asarray([x+[wordDict['<PAD>']]*(answer_maxlen-len(x)) for x in answers])
                answer_targets = [x+[wordDict['<PAD>']]*(answer_maxlen-len(x)) for x in answer_targets]
                answer_lens = np.repeat(answer_maxlen, len(answers))
                '''
                # To avoid training correlated tokens.

                for i in np.arange(answer_maxlen):
                    cnt_dict = dict()
                    for j in np.arange(len(answer_targets)):
                        token = str(answer_targets[j][i])
                        if token in cnt_dict:
                            cnt_dict[token][1].append(j)
                            if cnt_dict[token][0] < r_mask[j][i]:
                                cnt_dict[token][0] = r_mask[j][i]
                        else:
                            cnt_dict.update({token:[r_mask[j][i],[j]]})
                    for key, value in cnt_dict.items():
                        max_reward = value[0]
                        for j in value[1]:
                            if r_mask[j][i] != 0 and r_mask[j][i] < max_reward:
                                r_mask[j][i] = 0
                '''

                inputs_embeded = [[w2v.lookup(x) for x in y] for y in inputs]
                answers_embeded = [[w2v.lookup(x) for x in y] for y in answers]
                answer_targets_flatten = []
                for target in answer_targets:
                    answer_targets_flatten.extend(target)
                answer_targets_flatten = [[x, y] for x, y in enumerate(answer_targets_flatten)]
                sess_1[k].run(RI_optimizer_1[k],
                            feed_dict = {
                                inputs_1[k]: inputs_embeded,
                                outputs_1[k]: answers_embeded,
                                RI_targets_1[k]: answer_targets_flatten,
                                    masks_1[k]: r_mask,
                                    target_sequence_length_1[k]: answer_lens,
                                })
        else:
            # Pad the states.
            inputs_maxlen = max([len(x) for x in inputs])
            inputs = [[wordDict['<PAD>']]*(inputs_maxlen-len(x)) + x for x in inputs]
            r_mask = []

            # Train the network.
            # Generate a list of reward for each action.
            answer_maxlen = max([len(answer) for answer in answers])
            for i, answer in enumerate(answers):
                r_mask.append([rewards[i]*(-1)]*len(answer) + [0]*(answer_maxlen - len(answer)))
            # Pad the actions.
            answers = np.asarray([x+[wordDict['<PAD>']]*(answer_maxlen-len(x)) for x in answers])
            answer_targets = [x+[wordDict['<PAD>']]*(answer_maxlen-len(x)) for x in answer_targets]
            answer_lens = np.repeat(answer_maxlen, len(answers))
            '''
            for i in np.arange(answer_maxlen):
                cnt_dict = dict()
                for j in np.arange(len(answer_targets)):
                    token = str(answer_targets[j][i])
                    if token in cnt_dict:
                        cnt_dict[token][1].append(j)
                        if cnt_dict[token][0] < r_mask[j][i]:
                            cnt_dict[token][0] = r_mask[j][i]
                    else:
                        cnt_dict.update({token:[r_mask[j][i],[j]]})
                for key, value in cnt_dict.items():
                    max_reward = value[0]
                    for j in value[1]:
                        if r_mask[j][i] != 0 and r_mask[j][i] < max_reward:
                            r_mask[j][i] = 0
            '''

            inputs_embeded = [[w2v.lookup(x) for x in y] for y in inputs]
            answers_embeded = [[w2v.lookup(x) for x in y] for y in answers]
            answer_targets_flatten = []
            for target in answer_targets:
                answer_targets_flatten.extend(target)
            answer_targets_flatten = [[x, y] for x, y in enumerate(answer_targets_flatten)]
            sess_1[0].run(RI_optimizer_1[0],
                        feed_dict = {
                            inputs_1[0]: inputs_embeded,
                            outputs_1[0]: answers_embeded,
                            RI_targets_1[0]: answer_targets_flatten,
                                masks_1[0]: r_mask,
                                target_sequence_length_1[0]: answer_lens,
                            })


        # Store the model and log-----------------------------------------------
        if cnt % 10 == 0:
            print('Total reward for batch_{}: {}'.format(cnt, batch_reward))
        if cnt % 20 == 0:
            print(rewards)
            rewards_log.append(batch_reward)
            saver_1_1.save(sess_1[0], './models/RI_model')
            np.savetxt('./logs/RI_history.csv', rewards_log, delimiter = ',')
            print('Model and log are saved!')
