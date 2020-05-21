import codecs
import collections
import os
import pickle

import numpy as np
import tensorflow as tf

from nlpgnn.tokenizers import tokenization


class TFWriter(object):
    def __init__(self, maxlen, vocab_files, modes, task="NER", do_low_case=True,
                 check_exist=False, tokenizer="wordpiece", spm_model_file=None):
        self.maxlen = maxlen
        if tokenizer == "wordpiece":
            self.fulltoknizer = tokenization.FullTokenizer.bert_scratch(
                vocab_file=vocab_files, do_lower_case=do_low_case
            )
            self.convert_to_unicode = tokenization.convert_to_unicode

        elif tokenizer == "sentencepiece":
            self.fulltoknizer = tokenization.FullTokenizer.albert_scratch(
                vocab_file=vocab_files, do_lower_case=do_low_case, spm_model_file=spm_model_file
            )
            self.convert_to_unicode = tokenization.convert_to_unicode
        if modes != None:
            for mode in modes:
                self.mode = mode
                print("Writing {}".format(self.mode))
                self.filename = os.path.join("Input", self.mode)
                if task == "NER":
                    self.write(mode, check_exist, task)
                elif task == "cls":
                    self.write(mode, check_exist, task)

    def write(self, mode, check_exist, task):
        if check_exist:
            if os.path.exists(self.filename + ".tfrecords"):
                self.label_map = pickle.load(open(os.path.join("Input", "label2id.pkl"), 'rb'))
                print("Having Writen {} file in to device successfully!".format(mode))
                pass
            else:
                examples = self._read_file()
                self.label_map = self.label2id()
                self._write_examples(examples, task)
        else:
            examples = self._read_file()
            self.label_map = self.label2id()
            self._write_examples(examples, task)

    def _read_file(self):
        with codecs.open(self.filename, encoding='utf-8') as rf:
            examples = self._creat_examples(rf, self.mode)
        return examples

    def _creat_examples(self, lines, mode):
        examples = []
        self.label_list = set()
        for line in lines:
            line = line.strip().split('\t')
            # if mode == "test":
            #     w = self.convert_to_unicode(line[0])
            #     label = "0"
            # else:
            w = self.convert_to_unicode(line[0])
            label = self.convert_to_unicode(line[-1])
            examples.append((w, label))
            self.label_list.update(set(label.split()))
        self.label_list = sorted(self.label_list)
        print("Totally use {} labels!\n".format(len(self.label_list)))
        return examples

    def _creat_features(self, values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    def label2id(self):
        label_map = {}
        for (i, label) in enumerate(self.label_list):
            label_map[label] = i
        pickle.dump(label_map, open(os.path.join("Input", "label2id.pkl"), 'wb'))
        return label_map

    def _convert_single_example_ner(self, example, maxlen):
        # self.label_map =

        tokens = ["[CLS]"]
        segment_ids = [0]
        input_mask = [1]
        label_id = [self.label_map.get("O")]
        sentences, labels = example
        labels = labels.split()
        sentences = [self.fulltoknizer.tokenize(w) for w in sentences.split()]
        for i, words in enumerate(sentences):
            for word in words:
                if len(tokens) < maxlen - 1:
                    tokens.append(word)
                    segment_ids.append(0)
                    input_mask.append(1)
                    label_id.append(self.label_map.get(labels[i]))
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        label_id.append(self.label_map.get("O"))
        input_ids = self.fulltoknizer.convert_tokens_to_ids(tokens)
        while len(input_ids) < maxlen:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_id.append(self.label_map.get("O"))
        return input_ids, segment_ids, input_mask, label_id

    def _convert_single_example_cls(self, example, maxlen):
        tokens = ["[CLS]"]
        segment_ids = [0]
        input_mask = [1]
        sentences, label = example
        sentences = self.fulltoknizer.tokenize(sentences)

        for i, word in enumerate(sentences):
            if len(tokens) < maxlen - 1:
                tokens.append(word)
                segment_ids.append(0)
                input_mask.append(1)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_mask.append(1)
        input_ids = self.fulltoknizer.convert_tokens_to_ids(tokens)
        while len(input_ids) < maxlen:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        return input_ids, segment_ids, input_mask, [int(label)]

    def _write_examples(self, examples, task="ner"):
        writer = tf.io.TFRecordWriter(os.path.join("Input", self.mode + ".tfrecords"))
        features = collections.OrderedDict()
        for example in examples:
            if task.lower() == "ner":
                input_ids, segment_ids, input_mask, label_id = self._convert_single_example_ner(example, self.maxlen)
            elif task.lower() == 'cls':
                input_ids, segment_ids, input_mask, label_id = self._convert_single_example_cls(example, self.maxlen)
            features["input_ids"] = self._creat_features(input_ids)
            features["label_id"] = self._creat_features(label_id)
            features["segment_ids"] = self._creat_features(segment_ids)
            features["input_mask"] = self._creat_features(input_mask)

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
        writer.close()

    def convert_id_to_vocab(self, items):
        vocabs = self.fulltoknizer.convert_ids_to_tokens(items)
        return vocabs

    def convert_id_to_label(self, items):
        id2label = {value: key for key, value in self.label_map.items()}
        output = []
        for item in items:
            output.append(id2label[item])
        return output

    def get_init_weight(self, word2vec, vocab_size, embedding_size):
        init_weights = np.zeros([vocab_size, embedding_size])
        with codecs.open(word2vec, encoding='utf-8') as rf:
            line = rf.readline()
            embed = int(line.strip().split()[-1])
            if embedding_size != embed:
                print(embedding_size, embed)
                raise ValueError("embedding_size must equal word2vec size!")
            for line in rf:
                lines = line.strip().split()
                word_index = self.fulltoknizer.convert_tokens_to_ids([lines[0]])
                embedding = lines[1:]
                init_weights[word_index[0]] = embedding
        return list(init_weights)


class TFLoader(object):
    def __init__(self, maxlen, batch_size, task="ner", epoch=None):
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.epoch = epoch
        self.task = task

    def decode_record(self, record):
        # 告诉解码器每一个feature的类型
        if self.task.lower() == 'ner':

            feature_description = {
                "input_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "label_id": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "input_mask": tf.io.FixedLenFeature([self.maxlen], tf.int64),

            }

        elif self.task.lower() == "cls":
            feature_description = {
                "input_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "label_id": tf.io.FixedLenFeature([], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([self.maxlen], tf.int64),
                "input_mask": tf.io.FixedLenFeature([self.maxlen], tf.int64),

            }

        example = tf.io.parse_single_example(record, feature_description)
        return example["input_ids"], example["segment_ids"], example["input_mask"], example["label_id"]

    def load_train(self):
        self.filename = os.path.join("Input", "train.tfrecords")
        raw_dataset = tf.data.TFRecordDataset(self.filename)
        dataset = raw_dataset.map(map_func=lambda record: self.decode_record(record))
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat(self.epoch)

        dataset = dataset.batch(batch_size=self.batch_size,
                                drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def load_valid(self):
        self.filename = os.path.join("Input", "valid.tfrecords")
        raw_dataset = tf.data.TFRecordDataset(self.filename)
        dataset = raw_dataset.map(
            lambda record: self.decode_record(record)
        )

        dataset = dataset.batch(
            batch_size=self.batch_size,
            drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
