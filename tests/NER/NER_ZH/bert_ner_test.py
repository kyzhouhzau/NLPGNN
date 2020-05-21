import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
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

    def call(self, inputs, is_training=True):
        bert = self.bert(inputs, is_training)
        sequence_output = bert.get_sequence_output()  # batch,sequence,768
        output = self.dense(sequence_output)
        output = tf.reshape(output, [self.batch_size, self.maxlen, -1])
        output = tf.math.softmax(output, axis=-1)
        return output

    def predict(self, inputs, is_training=False):
        predict = self(inputs, is_training=is_training)
        return predict


model = BERT_NER(param)

model.build(input_shape=(3, param.batch_size, param.maxlen))

model.summary()

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                  modes=["valid"], check_exist=True)

ner_load = TFLoader(param.maxlen, param.batch_size)

# Metrics
f1score = Metric.SparseF1Score("macro")
precsionscore = Metric.SparsePrecisionScore("macro")
recallscore = Metric.SparseRecallScore("macro")
accuarcyscore = Metric.SparseAccuracy()

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
    predict = model.predict([X, token_type_id, input_mask])  # [batch_size, max_length,label_size]
    # predict = tf.argmax(output, -1)
    f1s.append(f1score(Y, predict))
    precisions.append(precsionscore(Y, predict))
    recalls.append(recallscore(Y, predict))
    accuracys.append(accuarcyscore(Y, predict))

print("f1:{}\tprecision:{}\trecall:{}\taccuracy:{}\n".format(np.mean(f1s),
                                                             np.mean(precisions),
                                                             np.mean(recalls),
                                                             np.mean(accuracys)))
# print("Sentence", writer.convert_id_to_vocab(tf.reshape(X,[-1]).numpy()))
#
# print("Label", writer.convert_id_to_label(tf.reshape(predict,[-1]).numpy()))
