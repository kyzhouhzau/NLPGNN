#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import codecs
import logging
import multiprocessing
import os
import re

import bz2file
from gensim.corpora.wikicorpus import extract_pages, filter_wiki
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from tqdm import tqdm

from nlpgnn.tokenizers import tokenization


class BPE(object):
    def __init__(self,corpus, vocab_files,language='zh', do_low_case=True):
        self.corpus = corpus
        self.fulltoknizer = tokenization.FullTokenizer(
            vocab_file=vocab_files, do_lower_case=do_low_case
        )
        if os.path.basename(corpus).endswith(".bz2"):
            self.wiki_bz_process(language)
        else:
            pass
        self.corpus = os.path.join(os.path.dirname(self.corpus), 'wiki.txt')
        if not os.path.exists(os.path.join(os.path.dirname(self.corpus), "pre_" + os.path.basename(self.corpus))):
            self.process_corpus()
        self.processed = os.path.join(os.path.dirname(self.corpus), "pre_" + os.path.basename(self.corpus))

    def wiki_bz_process(self,language):
        wiki = extract_pages(bz2file.open(self.corpus))
        f = codecs.open(os.path.join(os.path.dirname(self.corpus), 'wiki.txt'),
                        'w', encoding='utf-8')
        w = tqdm(wiki, desc="Currently get 0 files!")
        if language=='zh':
            for i, d in enumerate(w):
                if not re.findall('^[a-zA-Z]+:', d[0]) and not re.findall(u'^#', d[1]):
                    s = self.wiki_replace(d)
                    f.write(s + '\n\n\n')
                    i += 1
                    if i % 100 == 0:
                        w.set_description('Currently got %s files' % i)
        elif language=='en':
            pass

    def wiki_replace(self, d):
        s = d[1]
        s = re.sub(':*{\|[\s\S]*?\|}', '', s)
        s = re.sub('<gallery>[\s\S]*?</gallery>', '', s)
        s = re.sub('(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
        s = filter_wiki(s)
        s = re.sub('\* *\n|\'{2,}', '', s)
        s = re.sub('\n+', '\n', s)
        s = re.sub('\n[:;]|\n +', '\n', s)
        s = re.sub('\n==', '\n\n==', s)
        s = u'【 ' + d[0] + u' 】\n' + s
        return s

    def process_corpus(self):
        output = os.path.join(os.path.dirname(self.corpus), "pre_" + os.path.basename(self.corpus))
        wf = codecs.open(output, 'w', encoding='utf-8')
        with codecs.open(self.corpus, encoding='utf-8') as rf:
            for line in rf:
                sentence = tokenization.convert_to_unicode(line)
                sentences = self.fulltoknizer.tokenize(sentence)
                if len(sentences) != 0:
                    wf.write(" ".join(sentences) + '\n')
        wf.close()
        return output

    def train_word2vec(self,
                       embed_size=100,
                       window_size=5,
                       min_count=0,
                       iter=10):

        logger = logging.getLogger("bpe.py")

        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("running %s" % "bpe.py")
        output = os.path.join(os.path.dirname(self.corpus), "word2vec.vector")
        model = Word2Vec(LineSentence(self.processed),
                         size=embed_size, window=window_size, min_count=min_count, iter=iter,
                         workers=multiprocessing.cpu_count())
        model.save(os.path.join(os.path.dirname(self.corpus), "word2vec.model"))
        model.wv.save_word2vec_format(output, binary=False)
