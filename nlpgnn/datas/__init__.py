#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

import os
import importlib
# automatically import any Python files in the datas/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith(".py") and not file.startswith("_"):
#         dataset_name = file[: file.find(".py")]
#         module = importlib.import_module("nlpgnn.datas." + dataset_name)
from .checkpoint import *
from .dataloader import *

from .word2vec import *
from .graphloader import TuckERLoader, GCNLoaderzero, RGCNLoader, Planetoid, TUDataset, Sminarog
