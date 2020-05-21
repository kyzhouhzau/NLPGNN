import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from bert import BERT
import tqdm
from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter
from nlpgnn.tools import bert_init_weights_from_checkpoint
import numpy

# 载入参数
load_check = LoadCheckpoint(language='en', cased=False)
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
train_file = "data/R8/r8-train-all-terms.txt"
test_file = "data/R8/r8-test-all-terms.txt"
param.batch_size = 1
param.maxlen = 512

writer = TFWriter(param.maxlen, vocab_file, modes=None)


def target2index(input_file):
    label2index = {}
    node2index = {"<UNK>": 0}
    with open(input_file) as rf:
        for line in rf:
            line = line.strip().split('\t')
            sentences, label = line[1], line[0]
            word_set = set(sentences.split(' '))
            if label not in label2index:
                label2index[label] = len(label2index)
            for w in word_set:
                if w not in node2index:
                    node2index[w] = len(node2index)

    return label2index, node2index


def load_inputs(input_file, maxlen, label2index, num=5485, batch_size=10):
    input_ids = []
    segment_ids = []
    input_masks = []
    labels = []
    pbar = tqdm.tqdm(total=num)
    with open(input_file) as rf:
        for line in rf:
            line = line.strip().split('\t')
            sentences, label = line[1], line[0]
            label = label2index[label]
            tokens = ["[CLS]"]
            segment_id = [0]
            input_mask = [1]

            sentences = [writer.fulltoknizer.tokenize(w) for w in sentences.split()]
            for i, words in enumerate(sentences):
                for word in words:
                    if len(tokens) < maxlen - 1:
                        tokens.append(word)
                        segment_id.append(0)
                        input_mask.append(1)
            # tokens.append("[SEP]")
            # segment_ids.append(0)
            # input_mask.append(1)
            input_id = writer.fulltoknizer.convert_tokens_to_ids(tokens)
            while len(input_id) < maxlen:
                input_id.append(0)
                input_mask.append(0)
                segment_id.append(0)
            if len(input_ids) < batch_size:
                input_ids.append(input_id)
                segment_ids.append(segment_id)
                input_masks.append(input_mask)
                labels.append(label)
            else:
                yield tf.constant(input_ids, dtype=tf.int32), \
                      tf.constant(segment_ids, dtype=tf.int32), \
                      tf.constant(input_masks, dtype=tf.int32), \
                      labels
                input_ids = []
                segment_ids = []
                input_masks = []
                labels = []
            pbar.update(batch_size)


def build_sentence_graph(batch_sentence, batch_attention_matrix, node2index, top_k=5):
    word2indexs = []
    for sentence in batch_sentence:
        word2index = {}
        for w in sentence:
            if w not in word2index:
                word2index[w] = len(word2index)
        word2indexs.append(word2index)
    index2words = [{value: key for key, value in word2index.items()} for word2index in word2indexs]

    batch_nodes = [[index2word[i] for i in range(len(index2word))] for k, index2word in enumerate(index2words)]
    batch_value_index = [tf.math.top_k(attention_matrix, k=top_k) for attention_matrix in batch_attention_matrix]
    batch_edge_lists = []
    batch_edge_weights = []
    for i, sentence in enumerate(batch_sentence):
        edge_lists = []
        edge_weights = []
        for k, source in enumerate(sentence):
            for target_index in batch_value_index[i][1][k]:
                target = sentence[target_index.numpy()]
                edge_lists.append([source, target])
            edge_weights.extend([item.numpy() for item in batch_value_index[i][0][k]])
        batch_edge_lists.append(edge_lists)
        batch_edge_weights.append(edge_weights)
    batch_nodes = [[node2index.get(w, 0) for w in nodes] for i, nodes in
                   enumerate(batch_nodes)]
    batch_edge_lists = [[[word2indexs[i][edge_pair[0]], word2indexs[i][edge_pair[1]]] for edge_pair in edge_list] for
                        i, edge_list in enumerate(batch_edge_lists)]
    return batch_nodes, batch_edge_lists, batch_edge_weights


def draw_plot(matrix, x_label, y_label):
    sns.set()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, ax=ax)
    # 设置坐标字体方向
    plt.xticks(range(len(x_label)), x_label, rotation=90)
    plt.yticks(range(len(y_label)), x_label, rotation=360)
    plt.savefig("attention.png")
    plt.show()


# 构建模型
class BERT_NER(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(BERT_NER, self).__init__(**kwargs)
        self.batch_size = param.batch_size
        self.maxlen = param.maxlen
        self.label_size = param.label_size
        self.bert = BERT(param)

    def call(self, inputs, is_training=False):
        bert = self.bert(inputs, is_training)
        attention = bert.get_attention_matrix()[-1]  # T*T
        result = tf.reduce_sum(attention, 1)
        return result


model = BERT_NER(param)

model.build(input_shape=(3, param.batch_size, param.maxlen))

model.summary()

bert_init_weights_from_checkpoint(model, model_path, param.num_hidden_layers, pooler=False)


def build_trunk(model, input_file, label2index, node2index, num, mode="train",drw_attention=True):
    nodes = []
    labels_all = []
    edge_lists = []
    edge_weights = []
    batchs = []

    count = 0
    for input_ids, segment_ids, input_masks, labels in load_inputs(input_file, param.maxlen, label2index, num,
                                                                   param.batch_size):

        batch_sentence = [writer.convert_id_to_vocab(item) for item in input_ids.numpy()]
        predict = model([input_ids, segment_ids, input_masks])  # [batch_size, max_length,label_size]
        batch_mask_length = tf.reduce_sum(input_masks, -1)
        batch_attention_matrix = [
            tf.slice([predict[i]], [0, 1, 1], [-1, batch_mask_length[i] - 1, batch_mask_length[i] - 1])
            for i in range(len(batch_mask_length))]
        batch_sentence = [batch_sentence[i][1:batch_mask_length[i]] for i in range(len(batch_mask_length))]
        batch_word_index_list = []
        batch_nsentence = []  # target sentence
        for sentence in batch_sentence:
            nsentence = []
            word_index_list = []
            recoder = set()
            for w in sentence:
                if w[:2] == "##":
                    word_index_list.append(len(recoder) - 1)
                    nw = nsentence[-1] + w[2:]
                    nsentence.pop()
                    nsentence.append(nw)
                else:
                    word_index_list.append(len(recoder))
                    recoder.add(len(recoder))
                    nsentence.append(w)
            batch_nsentence.append(nsentence)
            batch_word_index_list.append(word_index_list)

        batch_attention_score = [tf.math.segment_sum(batch_attention_matrix[i][0], batch_word_index_list[i]) for i in
                                 range(len(batch_word_index_list))]

        batch_attention_score = [tf.transpose(batch_attention_score[i], [1, 0]) for i in
                                 range(len(batch_attention_score))]
        batch_attention_score = [tf.math.segment_sum(batch_attention_score[i], batch_word_index_list[i]) for i in
                                 range(len(batch_word_index_list))]
        batch_attention_score = [tf.transpose(batch_attention_score[i], [1, 0]) for i in
                                 range(len(batch_attention_score))]  # target matrix
        batch_nodes, batch_edge_lists, batch_edge_weights = build_sentence_graph(batch_nsentence, batch_attention_score,
                                                                                 node2index,
                                                                                 top_k=5)

        batch_count = [[i + count] * len(batch_nodes[i]) for i in range(len(batch_nodes))]

        nodes.extend(batch_nodes)
        edge_lists.extend(batch_edge_lists)
        edge_weights.extend(batch_edge_weights)
        labels_all.extend(labels)
        batchs.extend(batch_count)
        count += len(batch_nodes)
        if drw_attention:
            draw_plot(batch_attention_score[0], batch_nsentence[0], batch_nsentence[0])
            exit()

    np.save("data/{}_nodes.npy".format(mode), nodes)
    np.save("data/{}_edge_lists.npy".format(mode), edge_lists)
    np.save("data/{}_edge_weights.npy".format(mode), edge_weights)
    np.save("data/{}_labels.npy".format(mode), labels_all)
    np.save("data/{}_batchs.npy".format(mode), batchs)


label2index, node2index = target2index(train_file)
np.save("data/R8/node2index.npy", node2index)
print("Convert train datas to Graph .......")
build_trunk(model, train_file, label2index, node2index, num=5485, mode="train")
print("Convert test datas to Graph .......")
build_trunk(model, test_file, label2index, node2index, num=2189, mode="test")

# node2index, edge2index
