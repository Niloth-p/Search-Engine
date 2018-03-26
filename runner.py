'''
Takes the queries and returns the top 10 ranked documents
'''

# pylint: disable=C0103
# pylint: disable=no-else-return

import os
import time
import re
import pickle
import math
from collections import defaultdict, OrderedDict
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

direc = r"Z:\Assignment1\Programs\ExtractedText"
N = 6043


def getStopwords(stopwordsFile):
    '''Gets the stopwords'''
    print("Getting stop words")
    f = open(stopwordsFile, 'r')
    stopwords = [line.rstrip() for line in f]
    sw = dict.fromkeys(stopwords)
    f.close()
    return sw


def cleanqueries(porter, sw):
    '''Cleans the queries' tokens'''
    inp = input("Enter query: ")
    start = time.time()
    inp = inp.lower()
    queries = re.sub(r'[^a-z0-9 ]', ' ', inp)
    queries = word_tokenize(queries)
    print('Cleaning queries')
    queries = [porter.stem(query) for query in queries]
    queries = [x for x in queries if x not in sw]
    queries = set(queries)
    return queries, start


def computescores(queries):
    '''Computes the tf and idf scores for all the docs,
    and writes the dictionary of scores: docID -> score,
    to a file'''
    docID = 1
    count = 0
    scores = defaultdict(int)
    tf = loader("tf")
    idf = loader("idf")
    for root, dirs, files in os.walk(direc):
        for _ in files:
            score = 0
            for query in queries:
                if len(queries) > 1:
                    idfscore = computeIDF(query, idf)
                else:
                    idfscore = 1
                tfscore, flag = computeTF(query, docID, tf, count)
                if flag != 0:
                    count = flag
                score = score + idfscore*tfscore
            scores[docID] = score
            docID = docID + 1
    print('scores ready')
    writeToFile("scores", scores)
    writeToHumanReadableFile("scores", scores)
    return scores, count


def loader(index):
    '''Loads the object written to a file,
    using the pickle module'''
    with open(index + '.txt', 'rb') as f:
        index = pickle.loads(f.read())
    return index


def computeIDF(query, idf):
    '''Computes the idf score of each query term'''
    print('Computing IDF')
    if query in idf:
        df = idf[query]
        return math.log(N/df)
    else:
        return 0


def computeTF(query, docID, tf, count):
    '''Computes the tf score'''
    if query in tf:
        if tf[query][docID-1] == 0:
            return 0, count
        else:
            count = count + 1
            return math.log(1 + tf[query][docID - 1]), count
    else:
        return 0, 0


def ranking(scores):
    '''ranks the docIDs based on their tf-idf scores'''
    ranks = OrderedDict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    print('Ranking scores')
    writeToFile("ranks", ranks)
    writeToHumanReadableFile("ranks", ranks)
    retrieved = 0
    for k, v in ranks.items():
        if ranks[k] > 0:
            retrieved = retrieved + 1
        else:
            break
    if retrieved > 10:
        docIDs = list(ranks)[:10]
    elif retrieved > 0:
        docIDs = list(ranks)[:retrieved]
    else:
        return 0
    getDocNames(docIDs)
    return retrieved


def writeToFile(filename, index):
    '''Writes an object to a file,
    using the pickle module'''
    print('Writing to file - ' + filename)
    with open(filename + '.txt', 'wb') as f:
        pickle.dump(index, f)


def writeToHumanReadableFile(filename, index):
    '''Writes an object to a file,
    by formatting it into a readable format'''
    with open(filename + 'readable.txt', 'w') as f:
        if isinstance(index, dict):
            for k, v in index.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')
        else:
            for line in index:
                f.write(str(line) + '\n\n')
    f.close()
    print("Written to readable file " + filename + "readable.txt")


def getDocNames(docIDs):
    '''Retrieves the docnames from the docIDs'''
    print('Getting the doc names')
    print(docIDs)
    docNames = []
    count = 0
    while count < len(docIDs):
        x = docIDs[count]
        n = 1
        done = 0
        for root, dirs, files in os.walk(direc):
            for fname in files:
                if n == x:
                    docNames.append(fname)
                    done = 1
                    #print("Got a doc x : " + str(x) + " n : " + str(n))
                    count = count + 1
                    n = n + 1
                else:
                    if done == 0:
                        n = n + 1
                    else:
                        continue
    print(docNames)
    writeToHumanReadableFile("docNames", docNames)


def computePR(count, retrieved):
    """Computes the precision and recall"""
    precision = count/N
    print("There are " + str(count) + " documents relevant to the query")
    if retrieved <= 10:
        print("Retrieved " + str(retrieved) + " documents")
    else:
        print("Retrieved 10 documents")
    print("Precision : " + str(precision))
    if retrieved < 10 and retrieved != 0:
        recall = 1
        print("Recall : " + str(recall))
    elif retrieved > 10:
        recall = 10/retrieved
        print("Recall : " + str(recall))


def main():
    '''main function'''
    porter = PorterStemmer()
    sw = getStopwords("stopwords.dat")
    QUERIES, start = cleanqueries(porter, sw)
    scores, count = computescores(QUERIES)
    retrieved = ranking(scores)
    computePR(count, retrieved)
    end = time.time()
    print("running time : " + str(end - start))


if __name__ == '__main__':
    main()
