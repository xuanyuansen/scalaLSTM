#coding=UTF-8
'''
Created on 2016年8月2日

@author: wangshuai
'''
import gensim,logging
import os
import sys
import wsgiref
import numpy as np
from cStringIO import StringIO
import nltk
import itertools

class SimpleSentences(object):
    def __init__(self, filename):
        self.filename = filename
        self.idx=0

    def __iter__(self):
        for line in open(self.filename, 'r'):
            try:
                self.idx += 1
                if self.idx % 10000 ==0:
                    print "iter: ",self.idx,line
                yield "START,{0},END".format(line.strip()).split(",")
            except:
                print "illegal sentence"
                yield [""]

def trainword2vectormodel(datapath='dataFile', size=128, alpha=0.02, window=5, workers=8, min_count=1):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = SimpleSentences(datapath) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=size, alpha=alpha, window=window, workers=workers, min_count=min_count)
    if not os.path.exists('./word2vec/'):
        os.mkdir('./word2vec/')
    model.save('./word2vec/Word2VecModel')

def array2string(word, model):
    s = StringIO()
    np.savetxt(s, model[word], fmt='%.15f', newline=",")
    return s.getvalue()

def getvectors(inputFile,outputFile):
    with open(outputFile, 'w') as output:
        with open(inputFile) as inputfile:
            wsgiref
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            model=gensim.models.Word2Vec.load('./word2vec/Word2VecModel')

            lines=inputfile.readlines()
            tokenized_sentences = [line.strip().split(',') for line in lines]
            #print  tokenized_sentences
            word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
            print "Found {0} unique words.".format( len(word_freq.items() ) )
            vec_dic = [ k for k,v in word_freq.iteritems()  if len(k)>1]
            print "START", model["START"]
            print "END", model["END"]
            output.writelines( "{0}\t{1}\n".format( "START", array2string("START", model) ) )
            output.writelines( "{0}\t{1}\n".format( "END", array2string("END", model) ) )

            for element in vec_dic:
                output.writelines( "{0}\t{1}\n".format( element, array2string(element, model) ) )
            pass

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    trainword2vectormodel(sys.argv[1])
    getvectors(sys.argv[1], sys.argv[2])
    pass
