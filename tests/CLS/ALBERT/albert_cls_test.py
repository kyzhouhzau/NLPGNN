import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
from nlpgnn.models import albert

# 载入参数
load_check = LoadCheckpoint(language='en', model="albert", parameters="base")
param, vocab_file, model_path, spm_model_file = load_check.load_albert_param()
# 定制参数
param.batch_size = 8
param.maxlen = 10
param.label_size = 2


# 构建模型
class ALBERT_NER(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(ALBERT_NER, self).__init__(**kwargs)
        self.batch_size = param.batch_size
        self.maxlen = param.maxlen
        self.label_size = param.label_size
        self.albert = albert.ALBERT(param)
        self.dense = tf.keras.layers.Dense(self.label_size, activation="relu")

    def call(self, inputs, is_training=True):
        albert = self.albert(inputs, is_training)
        sequence_output = albert.get_pooled_output()  # batch,768
        pre = self.dense(sequence_output)
        output = tf.math.softmax(pre, axis=-1)
        return output

    def predict(self, inputs, is_training=False):
        output = self(inputs, is_training=is_training)
        return output


model = ALBERT_NER(param)

model.build(input_shape=(3, param.batch_size, param.maxlen))

model.summary()

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,task='cls',
                  modes=["valid"], check_exist=False,
                  tokenizer="sentencepiece", spm_model_file=spm_model_file)

ner_load = TFLoader(param.maxlen, param.batch_size,task='cls', epoch=10)

# 训练模型
# 使用tensorboard
summary_writer = tf.summary.create_file_writer("./tensorboard")

# Metrics
f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# For train model
# For test model
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
