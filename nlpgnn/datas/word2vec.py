#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import os
import codecs
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors

def get_embeddings(word,*arr):
    return word,np.array(arr,dtype=np.float32)

def build_matrix(embeddings_index,vocab_index,emb_dim):
    vocab_size = len(vocab_index)
    embeddings_matrix = np.zeros((len(vocab_index), emb_dim))
    for word, i in tqdm(vocab_index.items()):
        if i >= vocab_size: continue
        try:
            embeding_vector = embeddings_index[word]
        except:
            embeding_vector = embeddings_index["unknown"]
        if embeding_vector is not None:
            embeddings_matrix[i] = embeding_vector
    return embeddings_matrix

def glove(vocab_index_path,emb_dim):
    """
    get all vocabs' embedding
    :param vocab:
    :param emb_dim:
    :return:
    """
    # lf = codecs.open(vocab_index_path)
    # vocab_index = pickle.load(lf)
    vocab_index = np.load(vocab_index_path).item()
    url ="./embedding/glove.6B.{}d.txt".format(emb_dim)
    glove_name= "glove.6B.{}d.txt".format(emb_dim)
    try:
        data = tf.keras.utils.get_file("glove",origin=url)
        dirname = os.path.dirname(data)
    except ValueError:
        dirname = "./embedding/"

    embeddings_index = dict(get_embeddings(*o.split(" ")) for o in
                            codecs.open(os.path.join(dirname,glove_name),
                                        encoding='utf-8'))
    return build_matrix(embeddings_index, vocab_index, emb_dim)

def fast_text(vocab_index_path,emb_dim):
    vocab_index = np.load(vocab_index_path)

    url = "./embedding/crawl-{}d-2M.vec".format(emb_dim)
    fasttext_name = "crawl-{}d-2M.vec".format(emb_dim)
    try:
        data = tf.keras.utils.get_file("fasttext", origin=url)
        dirname = os.path.dirname(data)
    except ValueError:
        dirname = "./embedding/"

    embeddings_index = dict(
        get_embeddings(*o.strip().split(" ")) for o in codecs.open(os.path.join(dirname,fasttext_name),
                                        encoding='utf-8') if len(o) > 100)
    fasttext_embedding_matrix = build_matrix(embeddings_index, vocab_index,emb_dim)
    return fasttext_embedding_matrix

def word2vec(vocab_index_path,emb_dim):
    vocab_index = np.load(vocab_index_path)

    url = "./embedding/word2vec{}d.bin.gz".format(emb_dim)
    word2vec_name = "word2vec{}d.bin.gz".format(emb_dim)
    try:
        data = tf.keras.utils.get_file("word2vec", origin=url)
        dirname = os.path.dirname(data)
    except ValueError:
        dirname = "./embedding/"
    word2vec = KeyedVectors.load_word2vec_format(os.path.join(dirname,word2vec_name), binary=True)
    embeddings_index = {}
    for i, vec in tqdm(enumerate(word2vec.wv.vectors)):
        embeddings_index[word2vec.wv.index2word[i]] = vec
    word2vec_embedding_matrix = build_matrix(embeddings_index,  vocab_index,emb_dim)
    del word2vec
    return word2vec_embedding_matrix

