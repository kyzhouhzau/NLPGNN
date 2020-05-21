#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
data preprocessing for 中文糖尿病标注数据集。
target:
        train.bio
        valid.bio
        test.bio
"""

import jieba
import string
import codecs
import glob
import os


def build_jiebaic(filename):
    words = set()
    wf = codecs.open("jiebadic.txt", 'w', encoding='utf-8')
    with codecs.open(filename, encoding='utf-8') as rf:
        text = rf.read()
    result = jieba.tokenize(text)
    for tk in result:
        words.add(tk[0])
    for i in words:
        wf.write("{}\n".format(i))
        pass


def cut_file(annotation, filename):
    full = {}
    with codecs.open(filename, encoding='utf-8') as rf:
        text = rf.read()
    result = jieba.tokenize(text)
    for tk in result:
        tag = annotation.get((tk[1], tk[2]), -1)
        if tag == -1:
            tag = "O"
        else:
            annotation.pop((tk[1], tk[2]))
        full[(tk[1], tk[2])] = (tk[0], tag)
    return annotation, full


def ann_rebuild(filename):
    jieba.load_userdict("jiebadic.txt")
    rf = codecs.open(filename, encoding='utf-8')
    annotation = {}
    for line in rf:
        if line.startswith("T"):
            word = line.strip().split('\t')[-1]
            type_offset = line.strip().split('\t')[1].split(' ')
            type = type_offset[0]
            start = int(type_offset[1])
            final_end = int(type_offset[-1])
            result = jieba.tokenize(word)
            for i, tk in enumerate(result):
                end = start + tk[2] - tk[1]
                if i == 0:
                    type0 = "B-" + type
                    annotation[(start, end)] = type0
                elif end == final_end:
                    type1 = "E-" + type
                    annotation[(start, end)] = type1
                else:
                    type2 = "I-" + type
                    annotation[(start, end)] = type2
                start = end
    rf.close()
    return annotation


def mysplit(sentence):
    punc = list(string.punctuation)
    punc.extend([' ', '\t', '\n', '\r'])
    temp = ''
    lst = []
    for c in sentence:
        if '\u4e00' <= c <= '\u9fff' or c in punc:
            if temp != '':
                lst.append(temp)
                temp = ''
            lst.append(c)
        else:
            temp += c
    return lst


def ann_process(annotation):
    annotationdic = {}
    rf = codecs.open(annotation, encoding='utf-8')
    for line in rf:
        if line.startswith("T"):
            word = line.strip().split('\t')[-1]
            type_offset = line.strip().split('\t')[1].split(' ')
            type = type_offset[0]
            start = int(type_offset[1])
            end = int(type_offset[-1])
            word_list = mysplit(word)
            for i, w in enumerate(word_list):
                end = start + len(word_list[i])

                if i == 0:
                    type0 = "B-" + type
                    annotationdic[(start, end)] = type0
                elif i == len(word_list) - 1:
                    type1 = "E-" + type
                    annotationdic[(start, end)] = type1
                else:
                    type2 = "I-" + type
                    annotationdic[(start, end)] = type2
                start = end
    rf.close()
    return annotationdic


def raw_precess(ann, raw):
    full = []
    with codecs.open(raw, encoding='utf-8') as rf:
        rf = rf.read()
        # words = list(rf)
        words = mysplit(rf)
        start = 0
        for i, w in enumerate(words):
            end = start + len(words[i])
            tag = ann.get((start, end), -1)
            if tag == -1:
                tag = "O"
            if w not in ["\n", " "]:
                full.append((w, start, end, tag))
            start = end

    return full


def convert2bioe(file_dictionary, output):
    wf = codecs.open(os.path.join("InputNER", output), 'w', encoding='utf-8')
    ann_files = glob.glob(file_dictionary + "\*.ann")
    # ann_files = ["中文糖尿病标注数据集\\0.ann"]
    for file in ann_files:
        basefile = os.path.basename(file).split('.')[0]
        rawfile = os.path.join(file_dictionary, str(basefile) + '.txt')
        ann = ann_process(file)
        full = raw_precess(ann, rawfile)
        for Ws in full:
            wf.write("{}\t{}\t{}\t{}\n".format(Ws[0], Ws[1], Ws[2], Ws[3]))
        wf.write('\n')


def convert2biosentence(biofile, sentenceoutput):
    rf = codecs.open(os.path.join("InputNER", biofile), encoding="utf-8")
    wf = codecs.open(os.path.join("InputNER", sentenceoutput), 'w', encoding='utf-8')
    sentence = []
    label = []
    w = ''
    tag = ''
    for line in rf:
        line = line.strip().split()
        if len(line) == 0:
            pass
        else:
            w = line[0]
            tag = line[-1]
        if w in ["。","!",",","?",":",";"] or len(line) != 4:
            sentence.append(w)
            label.append(tag)
            sen = ' '.join(sentence)
            lab = ' '.join(label)
            if len(sentence)>1:
                wf.write("{}\t{}\n".format(sen, lab))
            sentence = []
            label = []
        else:
            sentence.append(w)
            label.append(tag)

    rf.close()
    wf.close()


convert2bioe("中文糖尿病标注数据集", "train.txt")
convert2biosentence("train.txt", "train")

