# This file builds a Google word2vec model.

import numpy as np

import gensim
import os, csv

# This class parse the content in a directory to a 2D list of tokens.
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.cnt = 0

    def __iter__(self):
        for file in os.listdir(self.dirname):
            self.cnt += 1
            if self.cnt % 2318 == 0:
                print('{} scans are completed'.format(self.cnt/2318))
            file = open(self.dirname+file, 'r')
            reader = csv.reader(file)
            for row in reader:
                yield row

# Scan all the files in the Inputs directory.
sentences = MySentences('Inputs/')
# Build the word2vec model.
word2vec_model = gensim.models.Word2Vec(sentences, size = 100, min_count = 0)

# save the model.
word2vec_model.save('word2vec_dict.model')
