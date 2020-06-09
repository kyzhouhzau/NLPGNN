import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
from nlpgnn.models import bert
from sklearn.metrics import classification_report

# 载入参数
load_check = LoadCheckpoint(language='en')
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param.batch_size = 8
param.maxlen = 100
param.label_size = 9

def ner_evaluation(true_label: list, predicts: list, masks: list):
    all_predict = []
    all_true = []
    true_label = [tf.reshape(item, [-1]).numpy() for item in true_label]
    predicts = [tf.reshape(item, [-1]).numpy() for item in predicts]
    masks = [tf.reshape(item, [-1]).numpy() for item in masks]
    for i, j, m in zip(true_label, predicts, masks):
        index = np.argwhere(m == 1)
        all_true.extend(i[index].reshape(-1))
        all_predict.extend(j[index].reshape(-1))
    report = classification_report(all_true, all_predict, digits=4)# paramaters labels
    print(report)

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
        pre = self.dense(sequence_output)
        pre = tf.reshape(pre, [self.batch_size, self.maxlen, -1])
        output = tf.math.softmax(pre, axis=-1)
        return output

    def predict(self, inputs, is_training=False):
        output = self(inputs, is_training=is_training)
        return output


model = BERT_NER(param)

model.build(input_shape=(3, param.batch_size, param.maxlen))

model.summary()

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                    modes=["valid"], check_exist=False)

ner_load = TFLoader(param.maxlen, param.batch_size, epoch=3)

# Metrics
f1score = Metric.SparseF1Score(average="macro")
precsionscore = Metric.SparsePrecisionScore(average="macro")
recallscore = Metric.SparseRecallScore(average="macro")
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('./save'))
# For test model
Batch = 0
predicts = []
true_label = []
masks = []
for X, token_type_id, input_mask, Y in ner_load.load_valid():
    predict = model.predict([X, token_type_id, input_mask])  # [batch_size, max_length,label_size]
    predict = tf.argmax(predict, -1)
    predicts.append(predict)
    true_label.append(Y)
    masks.append(input_mask)
print(writer.label2id())
ner_evaluation(true_label, predicts, masks)
    # print("Sentence", writer.convert_id_to_vocab(tf.reshape(X,[-1]).numpy()))
    #
    # print("Label", writer.convert_id_to_label(tf.reshape(predict,[-1]).numpy()))
