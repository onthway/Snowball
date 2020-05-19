#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David S. Batista"
__email__ = "dsbatista@inesc-id.pt"

import sys
import codecs
import re

from gensim import corpora
import nltk
from gensim.models import TfidfModel
import functools
from Snowball.utils import timer

class VectorSpaceModel(object):
    @timer
    def __init__(self, sentences_file, stopwords):
        self.dictionary = None
        self.corpus = None
        """              
        f_sentences = codecs.open(sentences_file, encoding='utf-8')
        documents = list()
        count = 0
        print("Gathering sentences and removing stopwords")
        
        for line in f_sentences:
            line = re.sub('<[A-Z]+>[^<]+</[A-Z]+>', '', line)

            # remove stop words and tokenize
            document = [word for word in nltk.word_tokenize(line.lower()) if word not in stopwords]
            documents.append(document)
            count += 1
            if count % 10000 == 0:
                sys.stdout.write(".")

        f_sentences.close()
              
        """        
        def cleansent(line, stopwords):
            line = re.sub('<[A-Z]+>[^<]+</[A-Z]+>', '', line)
            # remove stop words and tokenize
            document = [word for word in nltk.word_tokenize(line.lower()) if word not in stopwords]
            return document

        with open(sentences_file, 'r') as f:
            sents = f.read()
            f.close()
        #map(functools.partial(cleansent, y=2), a)

        documents = re.split('\n', sents)
        documents = [*map(functools.partial(cleansent, stopwords=stopwords), documents)]
        #print('xxxxxxx',len(documents))
        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(text) for text in documents]
        self.tf_idf_model = TfidfModel(self.corpus)

        print(len(documents), "documents red")
        print(len(self.dictionary), " unique tokens")
