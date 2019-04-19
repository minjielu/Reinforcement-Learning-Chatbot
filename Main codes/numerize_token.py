# The file parse the original data files which contain word tokens to a Input and a Output folder
# contain numerized tokens.

import os
import pandas as pd
import numpy as np
import operator, csv

def parse(x):
    # If the token x is in the vocabulary.
    if x in wordDict:
        return [wordDict[x]]
    # If x is the plural or past tense of another token in the vocabulary.
    if (x[-1] == 's' or x[-1] == 'd') and x[:-1] in wordDict:
        return [wordDict[x[:-1]]]
    if (x[-2:] == 'es' or x[-2:] == 'ed') and x[:-2] in wordDict:
        return [wordDict[x[:-2]]]
    # If x is a combination of two tokens in the vocabulary.
    for i in np.arange(1,len(x)):
        if x[:i] in wordDict and x[i:] in wordDict:
            return [wordDict[x[:i]],wordDict[x[i:]]]
    # If x is a name.
    if names[x] >= 3:
        return [wordDict['<NAME>']]
    # Otherwise unknown.
    return [wordDict['<UNK>']]

def num_unknown(x):
    return np.sum(x == wordDict['<UNK>'])

vocabulary = np.loadtxt('Vocabulary.csv', dtype = object, delimiter = '###')
wordDict = dict(zip(vocabulary[1:], range(len(vocabulary) - 1)))

wordDict['<UNK>'] = len(vocabulary) - 1
wordDict['<PAD>'] = len(vocabulary)
wordDict['<START>'] = len(vocabulary) + 1
wordDict['<END>'] = len(vocabulary) + 2
wordDict['<NAME>'] = len(vocabulary) + 3
cnt = 0
maxlen = 20
#os.mkdir('Inputs_1')
#os.mkdir('Outputs_1')

for filename in os.listdir('data'):
    # If an unknown token appears more than 5 times, it's probably a name.
    names = {}
    with open('data/' + filename, 'r') as file:
        line = file.readline()[:-1]
        while line:
            words = line.split()
            for word in words:
                if word not in wordDict:
                    if word not in names:
                        names[word] = 1
                    else:
                        names[word] += 1
            line = file.readline()[:-1]

    # Parse string tokens to integer tokens according to the vocabulary.
    with open('data/' + filename, 'r') as file:
        csvName = filename.split('.')[0] + '.csv'
        with open('Inputs_1/' + csvName, 'w') as inputFile:
            inputWriter = csv.writer(inputFile, delimiter = ',')
            with open('Outputs_1/' + csvName, 'w') as outputFile:
                outputWriter = csv.writer(outputFile, delimiter = ',')
                cnt += 1
                if cnt % 100 == 0:
                    print('{}/2318 files are parsed.'.format(cnt))
                line = file.readline()[:-1] # Get rid of the '\n' symbol.
                nextline = file.readline()[:-1]
                while nextline:
                    words = line.split()
                    nextWords = nextline.split()
                    newInput = []
                    for x in words:
                        newInput.extend(parse(x))

                    # Abandon a sentence if there are more than 3 unknowns in it.
                    if len(newInput) > maxlen or num_unknown(newInput) >= 3:
                        line = nextline
                        nextline = file.readline()[:-1]
                        continue
                    newOutput = []
                    for x in nextWords:
                        newOutput.extend(parse(x))
                    if len(newOutput) > maxlen or num_unknown(newOutput) >= 3:
                        line = file.readline()[:-1]
                        nextline = file.readline()[:-1]
                        continue

                    # Write the two sentences into files.
                    inputWriter.writerow(newInput)
                    outputWriter.writerow(newOutput)
                    line = nextline
                    nextline = file.readline()[:-1]
