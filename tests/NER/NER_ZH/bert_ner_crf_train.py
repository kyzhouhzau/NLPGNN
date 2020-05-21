import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
from nlpgnn.metrics.crf import CrfLogLikelihood
from nlpgnn.models import bert
from nlpgnn.optimizers import optim
from nlpgnn.tools import bert_init_weights_from_checkpoint

# 载入参数
# LoadCheckpoint(language='zh', model="bert", parameters="base", cased=True, url=None)
# language: the language you used in your input data
# model: the model you choose,could be bert albert and gpt2
# parameters: can be base large xlarge xxlarge for albert, base medium large for gpt2, base large for BERT.
# cased: True or false, only for bert model.
# url: you can give a link of other checkpoint.
load_check = LoadCheckpoint(language='zh')
param, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
param.batch_size = 8
param.maxlen = 10
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

# 构建优化器

optimizer_bert = optim.AdamWarmup(learning_rate=2e-5,  # 重要参数
                                  decay_steps=10000,  # 重要参数
                                  warmup_steps=1000, )
optimizer_crf = optim.AdamWarmup(learning_rate=1e-3,
                                 decay_steps=10000,  # 重要参数
                                 warmup_steps=1000,
                                 )
#
# 初始化参数
bert_init_weights_from_checkpoint(model,
                                  model_path,
                                  param.num_hidden_layers,
                                  pooler=False)

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                  modes=["train"], check_exist=False)

ner_load = TFLoader(param.maxlen, param.batch_size, epoch=5)

# 训练模型
# 使用tensorboard
summary_writer = tf.summary.create_file_writer("./tensorboard")

# Metrics
f1score = Metric.SparseF1Score(average="macro", predict_sparse=True)
precsionscore = Metric.SparsePrecisionScore(average="macro", predict_sparse=True)
recallscore = Metric.SparseRecallScore(average="macro", predict_sparse=True)
accuarcyscore = Metric.SparseAccuracy(predict_sparse=True)

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory="./save",
                                     checkpoint_name="model.ckpt",
                                     max_to_keep=3)
# For train model
Batch = 0
for X, token_type_id, input_mask, Y in ner_load.load_train():
    with tf.GradientTape(persistent=True) as tape:
        loss, predict = model([X, token_type_id, input_mask, Y])

        f1 = f1score(Y, predict)
        precision = precsionscore(Y, predict)
        recall = recallscore(Y, predict)
        accuracy = accuarcyscore(Y, predict)
        if Batch % 101 == 0:
            print("Batch:{}\tloss:{:.4f}".format(Batch, loss.numpy()))
            print("Batch:{}\tacc:{:.4f}".format(Batch, accuracy))
            print("Batch:{}\tprecision{:.4f}".format(Batch, precision))
            print("Batch:{}\trecall:{:.4f}".format(Batch, recall))
            print("Batch:{}\tf1score:{:.4f}".format(Batch, f1))
            manager.save(checkpoint_number=Batch)

        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=Batch)
            tf.summary.scalar("acc", accuracy, step=Batch)
            tf.summary.scalar("f1", f1, step=Batch)
            tf.summary.scalar("precision", precision, step=Batch)
            tf.summary.scalar("recall", recall, step=Batch)

    grads_bert = tape.gradient(loss, model.bert.variables + model.dense.variables)
    grads_crf = tape.gradient(loss, model.crf.variables)
    optimizer_bert.apply_gradients(grads_and_vars=zip(grads_bert, model.bert.variables + model.dense.variables))
    optimizer_crf.apply_gradients(grads_and_vars=zip(grads_crf, model.crf.variables))
    Batch += 1
