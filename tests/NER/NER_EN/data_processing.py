#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""

import glob
import os


def one_file(file):
    base_dir = os.path.dirname(file)
    output = os.path.basename(file).split('.')[0]
    rf = open(file)
    wf = open(os.path.join(base_dir, output), 'w')
    sentences = []
    tags = []
    sentence = []
    tag = []
    for line in rf:
        line = line.strip().split()
        if len(line) != 0:
            w = line[0]
            t = line[-1]
            sentence.append(w)
            tag.append(t)
        if w in [".", "?", "!", ";"] and len(sentence)!=0:
            sentences.append(sentence)
            tags.append(tag)
            sentence = []
            tag = []
    for sentence, tag in zip(sentences, tags):
        wf.write("{}\t{}\n".format(" ".join(sentence), " ".join(tag)))


def processing(base_dir):
    files = glob.glob(os.path.join(base_dir, "*"))
    for file in files:
        one_file(file)


processing("Input")
