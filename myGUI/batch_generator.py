# This batch_generator class provides batches for training models.
# The word2vec_lookup class provides the word2vec model.

import os, csv, gensim

class batch_generator:
    # This class generates batched training data.
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.file_list = os.listdir(dir)
        self.iterator = 0 # a pointer to the next file to be read.
        self.cur_content = [] # a 2d list stores the content in a file.
        self.cur_index = 0 # a pointer to the next line to be read.
        self.batch_size = batch_size

    def load_file(self):
        # Load the next file
        self.cur_content = []
        file = open(self.dir+self.file_list[self.iterator],'r')
        reader = csv.reader(file)
        for row in reader:
            row = [int(x) for x in row]
            self.cur_content.append(row)
        self.iterator += 1
        self.cur_index = 0

    def next_batch(self):
        # Return the next batch.
        if self.iterator == len(self.file_list) and self.cur_index == len(self.cur_content):
            return False, []
        new_batch = []
        while len(new_batch) < self.batch_size and self.iterator < len(self.file_list):
            if self.cur_index == len(self.cur_content):
                self.load_file()
            next_index = min(self.cur_index+self.batch_size-len(new_batch), len(self.cur_content))
            new_batch += self.cur_content[self.cur_index:next_index]
            self.cur_index = next_index
        return True, new_batch

class word2vec_lookup:
    # This class implements a Google word2vec model as the input embedding layer for the Seq2Seq model.
    def __init__(self, model_path):
        self.word2vec_model = gensim.models.Word2Vec.load(model_path)


    def lookup(self, x):
        # Add two extra dimensions to the word2vec space to represent the four special symbols
	    # <PAD>, <START>, <END> and <NAME>
        if x == 10000: #<PAD>
            return [0]*100 + [1]*2
        if x == 10001: #<START>
            return [0]*100 + [-1]*2
        if x == 10002: #<END>
            return [0]*100 + [1,-1]
        if x == 10003: #<NAME>
            return [0]*100 + [-1,1]
        return self.word2vec_model[str(x)].tolist() + [0]*2
