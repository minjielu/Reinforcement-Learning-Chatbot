# The file parse the original data files which contain word tokens to a Input and a Output folder
# contain numerized tokens.

import os
import pandas as pd
import numpy as np
import operator, csv

# Load the vocabulary.
vocabulary = np.loadtxt('Vocabulary.csv', dtype = object, delimiter = '###')
wordDict = dict(zip(vocabulary[1:], range(len(vocabulary) - 1)))

wordDict['<UNK>'] = len(vocabulary) - 1
wordDict['<PAD>'] = len(vocabulary)
wordDict['<START>'] = len(vocabulary) + 1
wordDict['<END>'] = len(vocabulary) + 2
cnt = 0
maxlen = 20
#os.mkdir('Inputs')
#os.mkdir('Outputs')
#os.mkdir('Masks')

#
for filename in os.listdir('data'):
    with open('data/' + filename, 'r') as file:
        csvName = filename.split('.')[0] + '.csv'
        with open('Inputs/' + csvName, 'w') as inputFile:
            inputWriter = csv.writer(inputFile, delimiter = ',')
            with open('Outputs/' + csvName, 'w') as outputFile:
                outputWriter = csv.writer(outputFile, delimiter = ',')
                with open('Masks/' + csvName, 'w') as maskFile:
                    maskWriter = csv.writer(maskFile, delimiter = ',')
                    cnt += 1
                    if cnt % 100 == 0:
                        print('{}/2318 files are parsed.'.format(cnt))
                    line = file.readline()[:-1]
                    nextline = file.readline()[:-1]
                    while nextline:
                        words = line.split()
                        nextWords = nextline.split()
                        if len(words) > maxlen:
                            line = nextline
                            nextline = file.readline()[:-1]
                        elif len(nextWords) > maxlen:
                            line = file.readline()[:-1]
                            nextline = file.readline()[:-1]
                        else:
                            newInput = [wordDict[x] if x in wordDict else wordDict['<UNK>'] for x in words]
                            newInput = [wordDict['<PAD>']] * (maxlen - len(newInput)) + newInput
                            inputWriter.writerow(newInput)
                            newMask = [1] * (len(nextWords) + 1) + [0] * (maxlen - len(nextWords))
                            maskWriter.writerow(newMask)
                            newOutput = [wordDict['<START>']] + \
                                        [wordDict[x] if x in wordDict else wordDict['<UNK>'] for x in nextWords] + \
                                        [wordDict['<END>']]
                            newOutput = newOutput + [wordDict['<PAD>']] * (maxlen + 2 - len(newOutput))
                            outputWriter.writerow(newOutput)
                            line = nextline
                            nextline = file.readline()[:-1]
