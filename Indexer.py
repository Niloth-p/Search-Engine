"""
    Traverses through the docs in a directory,
    creating the idf and the tf tables,
    and writes them to files
"""

# pylint: disable=C0103

import os
import re
from collections import defaultdict, Counter
import pickle
import time
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

direc = r"Z:\Assignment1\Programs\ExtractedText"


def getStopwords(stopwordsFile):
    '''get stopwords from the stopwords file'''
    f = open(stopwordsFile, 'r')
    stopwords = [line.rstrip() for line in f]
    sws = dict.fromkeys(stopwords)
    f.close()
    return sws


def traversethroughdocs(tf, porter, sw, postingslist):
    '''Traverses through all the docs in the directory indexing them'''
    docID = 1
    for root, dirs, files in os.walk(direc):
        for fname in files:
            current_file = "%s%s%s" % (os.path.abspath(root), os.path.sep, fname)
            print(current_file)
            tokensindoc, tf = tokenizedoc(tf, porter, current_file, sw, docID)
            postingslist = insertinpostingslist(tokensindoc, docID, postingslist)
            docID = docID + 1
    return postingslist, tf


def tokenizedoc(tf, porter, current_file, sw, docID):
    '''Returns all the cleaned tokens within the given doc'''
    file = open(current_file, 'r', encoding="utf-8")
    tokensindoc = []
    lines = file.readlines()
    for line in lines:
        line = line.lower()
        line = re.sub(r'[^a-z0-9 ]', ' ', line)
        tokensinline = word_tokenize(line)
        tokensindoc.extend(tokensinline)
    tokensindoc, tf = cleantokensofeachdoc(tf, porter, tokensindoc, sw, docID)
    file.close()

    return tokensindoc


def cleantokensofeachdoc(tf, porter, tokensindoc, sw, docID):
    '''Cleans a list of tokens - stemming, removing repetition,
    and adds the cleaned tokens to the tf table'''
    #Stemming
    tokensindoc = [porter.stem(token) for token in tokensindoc]

    #Removing stop words
    tokensindoc = [x for x in tokensindoc if x not in sw]

    TFhelper = dict(Counter(tokensindoc))
    #TFHelper is a dictionary of tokens and their counts, in that doc,
    #basically its a single column in the tf matrix
    tf = addTotf(tf, TFhelper, docID)
    #Removing repetition
    tokensindoc = set(tokensindoc)

    return tokensindoc, tf


def insertinpostingslist(tokensindoc, docID, postingslist):
    """
    Inserts all the given tokens of each doc into the postings list,
    in a suitable format, for later retrieval
    """
    if any(postingslist):
        for token, term in [(token, term) for token in tokensindoc for term in postingslist.keys()]:
            if token == term:
                postingslist[term].append(docID)
    else:
        for token in tokensindoc:
            postingslist[token] = [docID]
    return postingslist


def addTotf(tf, TFhelper, docID):
    '''Adds tokens of each doc to the tf table'''
    newterm = 1
    for token in TFhelper.keys():
        for term in tf.keys():
            if term == token:
                newterm = 0
                tf[term].append(TFhelper[token])
        if newterm == 1:
            tf[token] = [TFhelper[token]]
    for term in tf.keys():
        if len(tf[term]) != docID:
            tf[term].append(0)
    return tf


def createIDF(postingslist):
    '''Creates the idf table from the postingslist'''
    idf = defaultdict(int)
    for term in postingslist.keys():
        idf[term] = len(postingslist[term])
    return idf


def writeToFile(filename, index):
    '''Writes the given index to a file, as an object,
    using the pickle module'''
    with open(filename + '.txt', 'wb') as f:
        pickle.dump(index, f)
    print("Written to file " + filename + ".txt")


def writeToHumanReadableFile(filename, index):
    """
    Writes the given index to a file,
    in a suitable format for human reference,
    unlike an object which cannot be read
    """
    with open(filename + 'readable.txt', 'w') as f:
        for k, v in index.items():
            f.write(str(k) + ' >>> ' + str(v) + '\n\n')
    f.close()
    print("Written to readable file " + filename + "readable.txt")


def main():
    '''main function, calls other functions'''
    porter = PorterStemmer()
    SW = getStopwords("stopwords.dat")
    postingslist = defaultdict(list)
    tf = defaultdict(list)
    start = time.time()
    postingslist, tf = traversethroughdocs(tf, porter, SW, postingslist)
    IDF = createIDF(postingslist)
    end = time.time()
    writeToFile("postingslist", postingslist)
    writeToHumanReadableFile("postingslist", postingslist)
    writeToFile("idf", IDF)
    writeToHumanReadableFile("idf", IDF)
    writeToFile("tf", tf)
    writeToHumanReadableFile("tf", tf)
    print("running time : " + str(end - start))


if __name__ == '__main__':
    main()
