import os, csv, gensim

class batch_generator:
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.file_list = os.listdir(dir)
        self.iterator = 0
        self.cur_content = []
        self.cur_index = 0
        self.batch_size = batch_size

    def load_file(self):
        self.cur_content = []
        file = open(self.dir+self.file_list[self.iterator],'r')
        reader = csv.reader(file)
        for row in reader:
            row = [int(x) for x in row]
            self.cur_content.append(row)
        self.iterator += 1
        self.cur_index = 0

    def next_batch(self):
        if self.iterator == len(self.file_list) and self.cur_index+self.batch_size >= len(self.cur_content):
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
    def __init__(self, model_path):
        self.word2vec_model = gensim.models.Word2Vec.load(model_path)


    def lookup(self, x):
        if x == 10000: #<PAD>
            return [0]*100 + [1]*2
        if x == 10001: #<START>
            return [0]*100 + [-1]*2
        if x == 10002: #<END>
            return [0]*100 + [1,-1]
        if x == 10003: #<NAME>
            return [0]*100 + [-1,1]
        return self.word2vec_model[str(x)].tolist() + [0]*2
