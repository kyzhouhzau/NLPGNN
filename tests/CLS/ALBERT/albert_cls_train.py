import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.metrics import Metric
from nlpgnn.models import albert
from nlpgnn.optimizers import optim
from nlpgnn.tools import albert_init_weights_from_checkpoint

# 载入参数
# LoadCheckpoint(language='zh', model="bert", parameters="base", cased=True, url=None)
# language: the language you used in your input data
# model: the model you choose,could be bert albert and gpt2
# parameters: can be base large xlarge xxlarge for albert, base medium large for gpt2, base large for BERT.
# cased: True or false, only for bert model.
# url: you can give a link of other checkpoint.
load_check = LoadCheckpoint(language='en', model="albert", parameters="large")
param, vocab_file, model_path, spm_model_file = load_check.load_albert_param()
# 定制参数
param.batch_size = 8
param.maxlen = 100
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

# batch_size增加k学习率也要增加k倍
# optimizer_bert = optim.AdamWarmup(learning_rate=3e-6,  # 重要参数
#                                   decay_steps= 0,  # 重要参数
#                                   warmup_steps= 0)  # 0.1
optimizer_bert = optim.AdamWarmup(learning_rate=1e-6,
                                    decay_steps=40000)  # 0.1

sparse_categotical_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 初始化参数
albert_init_weights_from_checkpoint(model,
                                    model_path,
                                    param.num_hidden_layers,
                                    pooler=True)

# 写入数据 通过check_exist=True参数控制仅在第一次调用时写入
writer = TFWriter(param.maxlen, vocab_file,
                  modes=["train"], check_exist=False,task='cls',
                  tokenizer="sentencepiece", spm_model_file=spm_model_file)

ner_load = TFLoader(param.maxlen, param.batch_size,task='cls', epoch=3)

# 训练模型
# 使用tensorboard
summary_writer = tf.summary.create_file_writer("./tensorboard")

# Metrics
accuarcyscore = Metric.SparseAccuracy()

# 保存模型
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory="./save",
                                     checkpoint_name="model.ckpt",
                                     max_to_keep=3)
# For train model
Batch = 0
for X, token_type_id, input_mask, Y in ner_load.load_train():
    with tf.GradientTape() as tape:
        predict = model([X, token_type_id, input_mask])
        loss = sparse_categotical_loss(Y, predict)
        accuracy = accuarcyscore(Y, predict)
        if Batch % 100 == 0:
            print("Batch:{}\tloss:{:.4f}".format(Batch, loss.numpy()))
            print("Batch:{}\tacc:{:.4f}".format(Batch, accuracy))
        if Batch % 10 == 0:
            manager.save(checkpoint_number=Batch)

        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=Batch)
            tf.summary.scalar("acc", accuracy, step=Batch)
    grads_bert = tape.gradient(loss, model.variables)
    optimizer_bert.apply_gradients(grads_and_vars=zip(grads_bert, model.variables))
    Batch += 1
