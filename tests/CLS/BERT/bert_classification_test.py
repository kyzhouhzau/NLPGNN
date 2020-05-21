import numpy as np
import tensorflow as tf
from nlpgnn.models import bert
from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric, Losess

# 载入参数
load_check = LoadCheckpoint()
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param.batch_size = 8
param.maxlen = 100
param.label_size = 9


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
        sequence_output = bert.get_pooled_output()  # batch,768
        pre = self.dense(sequence_output)
        output = tf.math.softmax(pre, axis=-1)
        return output

    def predict(self, inputs, is_training=False):
        output = self(inputs, is_training=is_training)
        return output


model = BERT_NER(param)

model.build(input_shape=(3, param.batch_size , param.maxlen ))

model.summary()

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                    modes=["valid"], task='cls', check_exist=True)

load = TFLoader(param.maxlen, param.batch_size, task='cls')

# Metrics
f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# For train model

Batch = 0
f1s = []
precisions = []
recalls = []
accuracys = []
for X, token_type_id, input_mask, Y in load.load_valid():
    predict = model.predict([X, token_type_id, input_mask])  # [batch_size, max_length,label_size]
    f1s.append(f1score(Y, predict))
    precisions.append(precsionscore(Y, predict))
    recalls.append(recallscore(Y, predict))
    accuracys.append(accuarcyscore(Y, predict))
print("f1:{}\tprecision:{}\trecall:{}\taccuracy:{}\n".format(np.mean(f1s),
                                                             np.mean(precisions),
                                                             np.mean(recalls),
                                                             np.mean(accuracys)))
