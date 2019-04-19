# This file generate the vocabulary.

import os
import pandas as pd
import numpy as np
import operator, csv

worddict = {}
cnt = 0
maxlen = 0
maxSen = ""
# Count the frequencies of occurrence of all words.
for filename in os.listdir('data'):
    with open('data/' + filename, 'r') as file:
        cnt += 1
        if cnt % 100 == 0:
            print('{}/2318 files are scanned.'.format(cnt))
        line = file.readline()[:-1]
        while line:
            words = line.split()
            if len(words) > maxlen:
                maxlen = len(words)
                maxSen = line
            for word in words:
                if word in worddict:
                    worddict[word] += 1
                else:
                    worddict[word] = 1
            line = file.readline()[:-1]

# Sort the words according to their frequency of occurrence.
sorteddict = sorted(worddict.items(), key = operator.itemgetter(1), reverse = True)
sorteddict = pd.DataFrame(sorteddict, columns = ['words', 'counts'])

# Save the vocabulary into a csv file.
sorteddict.loc[:10000, 'words'].to_frame().to_csv('Vocabulary.csv', index = False, quoting = csv.QUOTE_NONE, escapechar = ' ')
print('Longest sentence has {} tokens'.format(maxlen))
print(maxSen)
