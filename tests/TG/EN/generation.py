#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.models import gpt2
from nlpgnn.sample import samples
from nlpgnn.tokenizers import gpt2_tokenization
from nlpgnn.tools import gpt2_init_weights_from_checkpoint

# 载入参数
# LoadCheckpoint(language='zh', model="bert", parameters="base", cased=True, url=None)
# language: the language you used in your input data
# model: the model you choose,could be bert albert and gpt2
# parameters: can be base large xlarge xxlarge for albert, base medium large for gpt2, base large for BERT.
# cased: True or false, only for bert model.
# url: you can give a link of other checkpoint.
load_check = LoadCheckpoint(language='en', model="gpt2", parameters="large")
param, vocab_file, model_path, encoder_file = load_check.load_gpt2_param()

param.batch_size = 10

tokenizer = gpt2_tokenization.FullTokenizer(encoder_file, vocab_file)
start_token = tokenizer.convert_token_to_id("<|endoftext|>")


# 构建模型
class GenGPT2(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(GenGPT2, self).__init__(**kwargs)
        self.model = gpt2.GPT2(param)

    def call(self, inputs, past=None, is_training=True):
        out = self.model(inputs, past, is_training)
        return out

    def predict(self, inputs, past=None, is_training=False):
        return self(inputs, past, is_training)


model = GenGPT2(param)
model.build(input_shape=(param.batch_size, param.maxlen))
model.summary()

gpt2_init_weights_from_checkpoint(model, model_path, param.n_layer)

generated = 0
for _ in range(10):
    output = samples.sample_sequence(model, param, length=40, start_token=start_token,
                                     temperature=1, top_k=0, top_p=1)
    for i in range(param["batch_size"]):
        generated += param["batch_size"]
        text = tokenizer.convert_tokens_to_string(output[i].numpy())
        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
        print(text)
