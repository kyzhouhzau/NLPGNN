#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:Kaiyin Zhou
"""
def glob_add_pool(batch,num_nodes=None):

    size = batch[-1].item() + 1 if num_nodes is None else num_nodes
    return scatter_('add', x, batch, dim_size=size)