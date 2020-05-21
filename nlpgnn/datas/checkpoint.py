#!encoding:utf-8
import json
import os
import sys
import tarfile
import zipfile

import requests

from nlpgnn.tools import dict_to_object


class LoadCheckpoint(object):
    
    def __init__(self, language='zh', model="bert",
                 parameters="base", cased=True, url=None):
        """
        :param language: what kind of language you will processing
        :param model: bert or albert
        :param parameters: could choose base large for bert base large xlarge xxlarge for albert
        :param cased: use cased of uncased checkpoint
        :param url: your own checkpoint
        """
        self.lg = language
        if model == "bert":
            if language == "zh":
                url = "https://storage.googleapis.com/bert_models/" \
                      "2018_11_03/chinese_L-12_H-768_A-12.zip"
            elif language == "en":
                if parameters == "base":
                    if cased:
                        url = "https://storage.googleapis.com/bert_models/" \
                              "2018_10_18/cased_L-12_H-768_A-12.zip"
                    else:
                        url = "https://storage.googleapis.com/bert_models/" \
                              "2018_10_18/uncased_L-12_H-768_A-12.zip"
                elif parameters == "large":
                    if cased:
                        url = "https://storage.googleapis.com/bert_models/" \
                              "2018_10_18/cased_L-24_H-1024_A-16.zip"
                    else:
                        url = "https://storage.googleapis.com/bert_models/" \
                              "2018_10_18/uncased_L-24_H-1024_A-16.zip"
                else:
                    print("Other models still not support! But you could set url equal the checkpoint link.")
                    url = url

        elif model == "albert":
            if language == "en":
                if parameters == "base":
                    url = "https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz"
                elif parameters == "large":
                    url = "https://storage.googleapis.com/albert_models/albert_large_v2.tar.gz"
                elif parameters == "xlarge":
                    url = "https://storage.googleapis.com/albert_models/albert_xlarge_v2.tar.gz"
                elif parameters == "xxlarge":
                    url = "https://storage.googleapis.com/albert_models/albert_xxlarge_v2.tar.gz"
                else:
                    raise ValueError("Other models still not support!")

            elif language == "zh":
                raise ValueError("Currently not support load chinese model!")

        elif model == "gpt2":
            self.all_files = ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001',
                              'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']
            self.base_dir = ""
            self.size = 0
            if language == "en":
                if parameters == "base":
                    if not os.path.exists("gpt_base"):
                        os.mkdir("gpt_base")
                    self.gpt_base_dir = "gpt_base"
                    self.gpt_size = "117M"
                elif parameters == "medium":
                    if not os.path.exists("gpt_medium"):
                        os.mkdir("gpt_medium")
                    self.gpt_base_dir = "gpt_medium"
                    self.gpt_size = "345M"
                elif parameters == "large":
                    if not os.path.exists("gpt_large"):
                        os.mkdir("gpt_large")
                    self.gpt_base_dir = "gpt_large"
                    self.gpt_size = "774M"
                elif parameters == "xlarge":
                    if not os.path.exists("gpt_xlarge"):
                        os.mkdir("gpt_xlarge")
                    self.gpt_base_dir = "gpt_xlarge"
                    self.gpt_size = "1558M"

        if model in ["bert", "albert"]:
            self.url = url
            self.size = self.getsize(self.url)
            filename = url.split('/')[-1]
            if not os.path.exists(filename):
                open(filename, 'w').close()
            if os.path.getsize(filename) != self.size:
                print("Download and unzip: {}".format(filename))
                self.download(url, filename, self.size)
            if filename.endswith("zip"):
                self.unzip(filename)
            elif filename.endswith('gz'):
                self.ungz(filename)
        if model in ["gpt2", "gpt"]:
            for filename in self.all_files:
                self.url = "https://storage.googleapis.com/gpt-2/models/" + self.gpt_size + "/" + filename
                # if not os.path.exists(self.gpt_base_dir):
                self.size = self.getsize(self.url)
                if os.path.exists(os.path.join(self.gpt_base_dir, filename)):
                    if os.path.getsize(os.path.join(self.gpt_base_dir, filename)) != self.size:
                        print("\nFetching {} .....".format(filename))
                        self.download(self.url, os.path.join(self.gpt_base_dir, filename), self.size)
                else:
                    print("\nFetching {} .....".format(filename))
                    self.download(self.url, os.path.join(self.gpt_base_dir, filename), self.size)

    def getsize(self, url):
        try:
            r = requests.head(url)
            size = r.headers.get("Content-Length")
            return int(size)
        except:
            print("Failed Download! Please Check your network!")
            exit()

    def bar(self, num, total):
        rate = num / total
        rate_num = int(rate * 100)
        if rate_num == 100:
            r = '\r%s>%d%%\n' % ('=' * int(rate_num / 3), rate_num,)  # 控制等号输出数量，除以3,表示显示1/3
        else:
            r = '\r%s>%d%%' % ('=' * int(rate_num / 3), rate_num,)
        sys.stdout.write(r)
        sys.stdout.flush()

    def download(self, url, output, total_size, chunk_size=1024):
        num = 0
        print(url)
        response = requests.get(url, stream=True)
        with open(output, 'wb') as wf:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    wf.write(chunk)
                    if total_size < chunk_size:
                        num = total_size
                    else:
                        num += chunk_size
                    self.bar(num, total_size)

    def unzip(self, filename):
        zip_file = zipfile.ZipFile(filename)
        zip_file.extractall()

    def ungz(self, filename):
        t = tarfile.open(filename)
        t.extractall()

    def load_bert_param(self, pretraining=False):
        filename = self.url.split('/')[-1]
        config = "{}/{}".format(filename.split('.')[0], "bert_config.json")
        vocab_file = "{}/{}".format(filename.split('.')[0], "vocab.txt")
        model_path = "{}/{}".format(filename.split('.')[0], "bert_model.ckpt")
        bert_param = json.load(open(config, 'r'))
        if not pretraining and self.lg == 'zh':
            bert_param.pop("directionality")
            bert_param.pop("pooler_fc_size")
            bert_param.pop("pooler_num_attention_heads")
            bert_param.pop("pooler_num_fc_layers")
            bert_param.pop("pooler_size_per_head")
            bert_param.pop("pooler_type")
        if not pretraining and self.lg == 'en':
            pass
        bert_param = dict_to_object(bert_param)
        bert_param.batch_size = 10
        bert_param.maxlen = 100
        bert_param.label_size = 2
        return bert_param, vocab_file, model_path

    def load_albert_param(self, pretraining=False):
        filename = self.url.split('/')[-1]
        config = "{}/{}".format('_'.join(filename.split('_')[:2]), "albert_config.json")
        vocab_file = "{}/{}".format('_'.join(filename.split('_')[:2]), "30k-clean.vocab")
        model_path = "{}/{}".format('_'.join(filename.split('_')[:2]), "model.ckpt-best")
        spm_model_file = "{}/{}".format('_'.join(filename.split('_')[:2]), "30k-clean.model")
        albert_param = json.load(open(config, 'r'))
        if not pretraining:
            albert_param.pop("net_structure_type")
            albert_param.pop("gap_size")
            albert_param.pop("num_memory_blocks")
            albert_param.pop("down_scale_factor")
        albert_param = dict_to_object(albert_param)
        albert_param.batch_size = 32
        albert_param.maxlen = 80
        albert_param.label_size = 10
        return albert_param, vocab_file, model_path, spm_model_file

    def load_gpt2_param(self, pretraining=False):
        config = "{}/{}".format(self.gpt_base_dir, "hparams.json")
        vocab_file = "{}/{}".format(self.gpt_base_dir, "vocab.bpe")
        encoder_file = "{}/{}".format(self.gpt_base_dir, "encoder.json")
        model_path = "{}/{}".format(self.gpt_base_dir, "model.ckpt")
        gpt_param = json.load(open(config, 'r'))
        gpt_param = dict_to_object(gpt_param)
        gpt_param.batch_size = 1
        gpt_param.maxlen = 1024
        gpt_param.label_size = 10
        return gpt_param, vocab_file, model_path,encoder_file





