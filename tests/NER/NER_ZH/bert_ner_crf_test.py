import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
from nlpgnn.metrics.crf import CrfLogLikelihood
from nlpgnn.models import bert

# 载入参数
load_check = LoadCheckpoint(language='zh')
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param.batch_size = 16
param.maxlen = 100
param.label_size = 46


# 构建模型
class BERT_NER(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(BERT_NER, self).__init__(**kwargs)
        self.batch_size = param.batch_size
        self.maxlen = param.maxlen
        self.label_size = param.label_size
        self.bert = bert.BERT(param)
        self.dense = tf.keras.layers.Dense(self.label_size, activation="relu")
        self.crf = CrfLogLikelihood()

    def call(self, inputs, is_training=True):
        # 数据切分
        input_ids, token_type_ids, input_mask, Y = tf.split(inputs, 4, 0)
        input_ids = tf.cast(tf.squeeze(input_ids, axis=0), tf.int64)
        token_type_ids = tf.cast(tf.squeeze(token_type_ids, axis=0), tf.int64)
        input_mask = tf.cast(tf.squeeze(input_mask, axis=0), tf.int64)
        Y = tf.cast(tf.squeeze(Y, axis=0), tf.int64)
        # 模型构建
        bert = self.bert([input_ids, token_type_ids, input_mask], is_training)
        sequence_output = bert.get_sequence_output()  # batch,sequence,768
        predict = self.dense(sequence_output)
        predict = tf.reshape(predict, [self.batch_size, self.maxlen, -1])
        # 损失计算
        log_likelihood, transition = self.crf(predict, Y, sequence_lengths=tf.reduce_sum(input_mask, 1))
        loss = tf.math.reduce_mean(-log_likelihood)
        predict, viterbi_score = self.crf.crf_decode(predict, transition,
                                                     sequence_length=tf.reduce_sum(input_mask, 1))
        return loss, predict

    def predict(self, inputs, is_training=False):
        loss, predict = self(inputs, is_training)
        return predict


model = BERT_NER(param)

model.build(input_shape=(4, param.batch_size, param.maxlen))

model.summary()

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                    modes=["valid"], check_exist=True)

ner_load = TFLoader(param.maxlen, param.batch_size)

# Metrics
f1score = Metric.SparseF1Score("macro",predict_sparse=True)
precsionscore = Metric.SparsePrecisionScore("macro",predict_sparse=True)
recallscore = Metric.SparseRecallScore("macro",predict_sparse=True)
accuarcyscore = Metric.SparseAccuracy(predict_sparse=True)

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# For test model
# print(dir(checkpoint))
Batch = 0
f1s = []
precisions = []
recalls = []
accuracys = []
for X, token_type_id, input_mask, Y in ner_load.load_valid():
    predict = model.predict([X, token_type_id, input_mask, Y])  # [batch_size, max_length,label_size]

    f1s.append(f1score(Y, predict))
    precisions.append(precsionscore(Y, predict))
    recalls.append(recallscore(Y, predict))
    accuracys.append(accuarcyscore(Y, predict))
    # print("Sentence", writer.convert_id_to_vocab(tf.reshape(X, [-1]).numpy()))
    #
    # print("Label", writer.convert_id_to_label(tf.reshape(predict, [-1]).numpy()))
print("f1:{}\tprecision:{}\trecall:{}\taccuracy:{}\n".format(np.mean(f1s),
                                                             np.mean(precisions),
                                                             np.mean(recalls),
                                                             np.mean(accuracys)))