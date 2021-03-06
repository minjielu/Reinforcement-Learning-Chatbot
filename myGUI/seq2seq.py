# This file is an evaluator that will respond to the Submit button of the GUI.

import numpy as np
import tensorflow as tf
from batch_generator import word2vec_lookup
from build_model import Seq2Seq


class Evaluator:

    def __init__(self):
        # Read in the vocabulary.
        self.x_seq_length = 20
        vocabulary = np.loadtxt(
            'Vocabulary.csv', dtype=object, delimiter='###')
        self.wordDict = dict(zip(vocabulary[1:], range(len(vocabulary) - 1)))
        self.wordDict['<UNK>'] = len(vocabulary) - 1
        self.wordDict['<PAD>'] = len(vocabulary)
        self.wordDict['<START>'] = len(vocabulary) + 1
        self.wordDict['<END>'] = len(vocabulary) + 2
        self.wordDict['<NAME>'] = len(vocabulary) + 3
        # Read in the Google Word2Vec model.
        self.w2v = word2vec_lookup('word2vec_dict.model')
        nodes = 1000
        embed_size = 100
        maxlen = 20

        tf.reset_default_graph()
        # Read in the reinforcement learning model.
        ri_graph = tf.Graph()
        with ri_graph.as_default():
            ri_model = Seq2Seq()
            self.ri_inputs, self.ri_outputs, _, _, _, _, \
                self.ri_target_sequence_length, _, _, self.ri_training_logits, \
                _, _, _ = ri_model.build_model()
            ri_saver = tf.train.Saver()
            self.ri_sess = tf.Session(graph=ri_graph)
            # Load the most up to date model from file.
            ri_saver.restore(self.ri_sess, "./Models/RI_model")

        tf.reset_default_graph()

        # Read in the forward maximum likelihood model.
        forward_graph = tf.Graph()
        with forward_graph.as_default():
            forward_model = Seq2Seq()
            self.forward_inputs, self.forward_outputs, _, _, _, _, \
                self.forward_target_sequence_length, _, _, self.forward_training_logits, \
                _, _, _ = forward_model.build_model()
            forward_saver = tf.train.Saver()
            self.forward_sess = tf.Session(graph=forward_graph)
            # Load the most up to date model from file.
            forward_saver.restore(self.forward_sess, "./Models/forward_model")

        # Read in the backward maximum likelihood model.
        backward_graph = tf.Graph()
        with backward_graph.as_default():
            backward_model = Seq2Seq()
            self.backward_inputs, self.backward_outputs, _, _, _, _, \
                self.backward_target_sequence_length, _, _, self.backward_training_logits, \
                _, _, _ = backward_model.build_model()
            backward_saver = tf.train.Saver()
            self.backward_sess = tf.Session(graph=backward_graph)
            # Load the most up to date model from file.
            backward_saver.restore(
                self.backward_sess, "./Models/backward_model")

    def str2num(self, inputString):
        # This function parses the input sentence to the format that can be accepted by the Seq2Seq model
        # For example, all letters should be lower case letters. Punctuations should be a separate element
        # from its preceding word in the list.
        res = []
        symbolList = [',', '.', '?', '!']
        for s in inputString.split():
            s = s.lower()
            if s[-1] in symbolList and len(s) > 1:
            # Take out punctuations
                res.extend(
                    [self.wordDict[s[:-1]] if s[:-1] in self.wordDict else self.wordDict['<UNK>'],
                     self.wordDict[s[-1]]])
            elif len(s.split("'")) > 1:
            # Handle cases like I', You'
                s = s.split("'")
                s1 = s[0] + "'"
                s2 = s[1]
                res.extend(
                    [self.wordDict[s1] if s1 in self.wordDict else self.wordDict['<UNK>'],
                     self.wordDict[s1] if s1 in self.wordDict else self.wordDict['<UNK>']])
            else:
                res.append(
                    self.wordDict[s] if s in self.wordDict else self.wordDict['<UNK>'])
        return res

    def num2str(self, numList):
        # This function transfers the direct output of the Seq2Seq model to
        # more general written forms.
        res = ""
        symbolList = [',', '.', '?', '!']
        if len(numList[0]) > 1:
        # Capitalize the first letter of a sentence.
            res = numList[0][0].upper() + numList[0][1:]
        else:
            res = numList[0].upper()
        for i in np.arange(1, len(numList)):
            if numList[i] == 'i':
            # Handle I
                res += ' I'
            elif numList[i] == "i'":
            # Handle I'
                res += "I'"
            elif numList[i - 1][-1] == "'" or numList[i] in symbolList:
            # Handle cases like I'm
                res += numList[i]
            else:
                res += ' ' + numList[i]
        return res

    def rewards(self, state, action):
        # Provide the rewards for an action.
        dull_set = [
            "i don' t know what you' re talking about .", "i don' t know .", "you don' t know .",
            "you know what i mean .", "i know what you mean .", "you know what i' m saying . ", "you don' know anything ."]

        # Calculate the reward.
        action = np.asarray(action)
        action_target = action[:, 1:]
        action = action[:, :-1]
        action_as_input = action[:, 1:]

        # Ease of answering.
        r1 = 0
        action_as_input_embeded = [[self.w2v.lookup(x) for x in y]
                                   for y in action_as_input]
        for dull_answer in dull_set:
            dull_answer = np.asarray(
                [[self.wordDict['<START>']] + [self.wordDict[x] if x in self.wordDict else self.wordDict['<UNK>'] for x in dull_answer.split()]])
            dull_answer_embeded = [[self.w2v.lookup(x) for x in y]
                                   for y in dull_answer]
            logits = self.forward_sess.run(self.forward_training_logits,
                                           feed_dict={
                                               self.forward_inputs: action_as_input_embeded,
                                           self.forward_outputs: dull_answer_embeded,
                                           self.forward_target_sequence_length: np.asarray([len(dull_answer[0])])})
            its = np.exp(logits[0])
            sums = np.sum(its, axis=1)
            dull_answer = np.concatenate(
                (dull_answer[:, 1:], np.full((dull_answer.shape[0], 1), self.wordDict['<END>'])), axis=1)
            r1 += np.sum(np.log([its[i, j] / sums[i]
                         for i, j in enumerate(dull_answer[0], 0)])) / len(dull_answer[0])
        r1 /= len(dull_set)
        r1 = np.exp(r1)

        # Semantic Coherence.
        # Forward possibility.
        state_embeded = [[self.w2v.lookup(x) for x in y]
                         for y in [state]]
        action_embeded = [[self.w2v.lookup(x) for x in y] for y in action]
        forward_logits = self.forward_sess.run(self.forward_training_logits,
                                               feed_dict={
                                                   self.forward_inputs: state_embeded,
                                               self.forward_outputs: action_embeded,
                                               self.forward_target_sequence_length: np.asarray([len(action[0])])})
        forward_its = np.exp(forward_logits[0])
        forward_sums = np.sum(forward_its, axis=1)
        forward_logpro = np.sum(np.log([
                                forward_its[i, j] / forward_sums[i] for i, j in enumerate(action_target[0], 0)]))
        forward_logpro /= len(action_target[0])

        # Backward possibility.
        state_embeded = [[self.w2v.lookup(x) for x in y]
                         for y in [[self.wordDict['<START>']] + state]]
        backward_logits = self.backward_sess.run(self.backward_training_logits,
                                                 feed_dict={
                                                 self.backward_inputs: action_as_input_embeded,
                                                 self.backward_outputs: state_embeded,  # q1_embeded,
                                                 self.backward_target_sequence_length: np.asarray([len(state) + 1])})
        backward_its = np.exp(backward_logits[0])
        backward_sums = np.sum(backward_its, axis=1)
        backward_logpro = np.sum(
            np.log([backward_its[i, j] / backward_sums[i] for i, j in enumerate(state + [self.wordDict['<END>']], 0)]))
        backward_logpro /= len(state) + 1
        r3 = forward_logpro + backward_logpro
        r3 = np.exp(r3)

        return r1, r3

    def evaluate(self, chatHistory):
        # Compute the score and the recommendation.
        rec = [[0] * 3]
        score = 0
        message = ""

        if len(chatHistory) != 1:
            input = chatHistory[-2]
            input = [self.str2num(input)]
            output = [[self.wordDict['<START>']]]

            # Generate a response using the reinforcement model.
            for i in range(self.x_seq_length + 1):
                input_embeded = [
                        [self.w2v.lookup(x) for x in y] for y in input]
                output_embeded = [
                        [self.w2v.lookup(x) for x in y] for y in output]
                batch_logits = self.ri_sess.run(self.ri_training_logits,
                                             feed_dict={
                                                 self.ri_inputs: input_embeded,
                                                 self.ri_outputs: output_embeded,
                                                 self.ri_target_sequence_length: [len(output[0])]})

                prediction = batch_logits[0, -1].argmax(axis=-1)
                output = np.hstack([output, [[prediction]]])
                if prediction == self.wordDict['<END>']:
                    break

            # Get the rewards for the response generated by the reinforcement
            # model.
            num2Word = dict(
                zip(self.wordDict.values(), self.wordDict.keys()))
            rec = output
            rec_r1, rec_r3 = self.rewards(input[0], rec)

            # Get the rewards for the user's reply.
            user_reply = chatHistory[-1]
            user_r1, user_r3 = self.rewards(input[0], [[self.wordDict['<START>']]+self.str2num(user_reply)+[self.wordDict['<END>']]])

            score = int(rec_r1 * user_r3*100 / user_r1 / rec_r3)
            # If the r1 reward of the user's response is much less than the r1
            # reward of the recommendation.
            if user_r3 / rec_r3 < 0.4:
                message += "\nYour response is totally irrelevant."
            # If the r3 reward of the user's response is much less than the r3
            # reward of the recommendation.
            if rec_r1 / user_r1 < 0.4:
                message += "\nYour response is hard to answer."

        # Compute the reply of the AI agent.
        input = chatHistory[-1]
        input = [self.str2num(input)]
        output = [[self.wordDict['<START>']]]

        for i in range(self.x_seq_length + 1):
            input_embeded = [[self.w2v.lookup(x) for x in y] for y in input]
            output_embeded = [[self.w2v.lookup(x) for x in y] for y in output]
            batch_logits = self.ri_sess.run(self.ri_training_logits,
                                         feed_dict={self.ri_inputs: input_embeded,
                                                    self.ri_outputs: output_embeded,
                                                    self.ri_target_sequence_length: [len(output[0])]})

            prediction = batch_logits[0, -1].argmax(axis=-1)

            if prediction == self.wordDict['<END>']:
                break
            output = np.hstack([output, [[prediction]]])

        num2Word = dict(zip(self.wordDict.values(), self.wordDict.keys()))
        nextTurn = output[0]
        nextTurn = [num2Word[x] if x != 10003 else 'Minjie' for x in nextTurn[1:]]
        nextTurn = self.num2str(nextTurn)

        # Parse the recommendation from a vector of integers to a string.
        rec = [num2Word[x] if x != 10003 else 'Minjie' for x in rec[0][1:-1]]
        rec = self.num2str(rec)

        return str(score), rec, nextTurn, message
