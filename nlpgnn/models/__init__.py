#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

import os
import importlib
# automatically import any Python files in the models/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith(".py") and not file.startswith("_"):
#         dataset_name = file[: file.find(".py")]
#         module = importlib.import_module("nlpgnn.models." + dataset_name)
from .albert import *
from .bert import *
from .GAT import *
from .GCN import *
from .GIN import *
from .gpt2 import *
from .RGCN import *
from .TextCNN import *
from .tucker import *
from .GraphSage import *
from .TextGCN2019 import *