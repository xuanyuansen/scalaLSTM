#coding=UTF-8
'''
Created on 2016年8月2日

@author: wangshuai
'''
import gensim,logging
import os
import sys


class SimpleSentences(object):
    def __init__(self, filename):
        self.filename = filename
        self.idx=0

    def __iter__(self):
        for line in open(self.filename, 'r'):
            try:
                self.idx += 1
                if self.idx % 500 ==0:
                    print "iter: ",self.idx,line
                yield "START,{0},END".format(line).split(",")
            except:
                print "illegal sentence"
                yield [""]

def trainword2vectormodel(datapath='dataFile', size=128, alpha=0.02, window=5, workers=8):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = SimpleSentences(datapath) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, size=size, alpha=alpha, window=window, workers=workers)
    if not os.path.exists('./word2vec/'):
        os.mkdir('./word2vec/')
    model.save('./word2vec/Word2VecModel')

def getvectors(inputFile,outputFile):
    with open(outputFile) as output:
        with open(inputFile) as inputfile:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            model=gensim.models.Word2Vec.load('./word2vec/Word2VecModelCSV')

            lines=inputfile.readlines()
            print "START", model["START"]
            print "END", model["END"]

            idx = 0
            for line in lines:
                idx += 1
                try:
                    for element in line.split(","):
                        if idx % 500 ==0:
                            try:
                                output.writelines(str(model[element]))
                            except:
                                print "no key"
                    outputFile.writelines("\n")
                except:
                    print "exception"

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding("utf-8")
    trainword2vectormodel(sys.argv[1])
    getvectors(sys.argv[1], sys.argv[2])
    pass