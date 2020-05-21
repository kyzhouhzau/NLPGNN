#! encoding:utf-8
import glob
import os
import pickle as pkl
import sys
from collections import defaultdict
from six.moves import urllib
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from scipy import sparse
import zipfile
import codecs
from nlpgnn.gnn.utils import *

from sklearn.model_selection import StratifiedKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class TuckERLoader():
    def __init__(self, base_path="data", reverse=True):
        self.train_data = self.load_data(base_path, 'train.txt', reverse=reverse)
        self.valid_data = self.load_data(base_path, 'valid.txt', reverse=reverse)
        self.test_data = self.load_data(base_path, 'test.txt', reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)

        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)

        self.relations = self.train_relations + \
                         [i for i in self.valid_relations if i not in self.train_relations] + \
                         [i for i in self.test_relations if i not in self.train_relations]

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.realtion_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        del self.train_relations
        del self.test_relations
        del self.valid_relations

    def data_dump(self, data="train"):
        if data == "train":
            data = self.train_data
        elif data == "valid":
            data = self.valid_data
        elif data == "test":
            data = self.test_data
        data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        return er_vocab, er_vocab_pairs

    def load_data(self, base_path, data_type="train", reverse=False):
        data = []
        with open(os.path.join(base_path, data_type)) as rf:
            for line in rf:
                contents = line.strip().split()
                data.append(contents)
                if reverse:
                    data.append([contents[2], contents[1] + "_reverse", contents[0]])
            return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def get_batch(self, er_vocab, er_vocab_pairs, batch_size=32):

        #
        targets = [er_vocab[key] for key in er_vocab_pairs]

        def generator():
            for x, y in zip(er_vocab_pairs, targets):
                yield {'h_r': x, 't': y}

        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_types={'h_r': tf.int32, 't': tf.int32})
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes={'h_r': [None], 't': [None]},
                                       drop_remainder=True)
        return dataset

    def get_data_idxs(self, data):  # data could be self.train_data self.valid_data self.test_data
        data_idxs = [
            (self.entity_idxs[data[i][0]], self.realtion_idxs[data[i][1]],
             self.entity_idxs[data[i][2]]) for i in range(len(data))
        ]
        print("Number of data points: %d" % len(data_idxs))
        return data_idxs

    def get_er_vocab(self, data_index):
        er_vocab = defaultdict(list)
        for triple in data_index:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def target_convert(self, targets, batch_size, num_entities):
        targets_one_hot = np.zeros((batch_size, num_entities))
        for idx, tar in enumerate(targets):
            targets_one_hot[idx, tf.gather_nd(tar, tf.where(tar > 0))] = 1
        return tf.constant(targets_one_hot)


class GCNLoaderzero:
    def __init__(self, base_path="data", dataset="cora"):
        self.base_path = base_path
        self.dataset = dataset
        print("Loading {} dataset...".format(dataset))

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = (1 / rowsum).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sparse.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def convert_2_sparse_tensor(self, sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
        values = sparse_matrix.data
        shape = sparse_matrix.shape
        # indices = np.array([[row, col] for row, col in zip(sparse_matrix.row, sparse_matrix.col)], dtype=np.int64)
        indices = tf.constant([[row, col] for row, col in zip(sparse_matrix.row, sparse_matrix.col)], dtype=tf.int32)
        return indices
        # return tf.sparse.SparseTensor(indices, values, shape)

    def load(self):
        idx_features_labels = np.genfromtxt("{}/{}.content".format(self.base_path, self.dataset),
                                            dtype=np.dtype(str))
        features = sparse.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # 构建图
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/{}.cites".format(self.base_path, self.dataset),
                                        dtype=np.int32)
        # [[1,2],
        #  [22,23]]
        # N*2
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        # adj = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                         shape=(labels.shape[0], labels.shape[0]),
        #                         dtype=np.float32)

        # 构建对称邻接矩阵

        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #
        # features = self.normalize(features)
        # adj = self.normalize(adj + sparse.eye(adj.shape[0]))

        features = tf.constant(np.array(features.todense()))

        labels = tf.constant(np.where(labels)[1])
        adj = tf.constant(edges, dtype=tf.int32)

        # adj = self.convert_2_sparse_tensor(adj)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        return features, adj, labels, idx_train, idx_val, idx_test


class RGCNLoader(object):
    def __init__(self, base_path="data", dataset="FB15k"):
        self.base_path = base_path
        self.dataset = dataset
        self.file_path = os.path.join(base_path, dataset)

    def read_triplets(self, file_path, entity2id, relation2id):
        triplets = []

        with open(file_path) as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

        return np.array(triplets)

    def load_data(self):
        print("load data from {}".format(self.file_path))

        with open(os.path.join(self.file_path, 'entities.dict')) as f:
            entity2id = dict()

            for line in f:
                eid, entity = line.strip().split('\t')
                entity2id[entity] = int(eid)

        with open(os.path.join(self.file_path, 'relations.dict')) as f:
            relation2id = dict()

            for line in f:
                rid, relation = line.strip().split('\t')
                relation2id[relation] = int(rid)

        train_triplets = self.read_triplets(os.path.join(self.file_path, 'train.txt'), entity2id, relation2id)
        valid_triplets = self.read_triplets(os.path.join(self.file_path, 'valid.txt'), entity2id, relation2id)
        test_triplets = self.read_triplets(os.path.join(self.file_path, 'test.txt'), entity2id, relation2id)

        print('num_entity: {}'.format(len(entity2id)))
        print('num_relation: {}'.format(len(relation2id)))
        print('num_train_triples: {}'.format(len(train_triplets)))
        print('num_valid_triples: {}'.format(len(valid_triplets)))
        print('num_test_triples: {}'.format(len(test_triplets)))

        return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

    def sample_edge_uniform(self, n_triples, sample_size):
        """Sample edges uniformly from all the edges."""
        all_edges = np.arange(n_triples)
        return np.random.choice(all_edges, sample_size, replace=False)

    def negative_sampling(self, pos_samples, num_entity, negative_rate):
        size_of_batch = len(pos_samples)
        num_to_generate = size_of_batch * negative_rate
        neg_samples = np.tile(pos_samples, (negative_rate, 1))
        labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
        labels[: size_of_batch] = 1
        values = np.random.choice(num_entity, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values[subj]
        neg_samples[obj, 2] = values[obj]

        return np.concatenate((pos_samples, neg_samples)), labels

    def edge_normalization(self, edge_type, edge_index, num_entity, num_relation):
        from nlpgnn.abandoned.scatter import scatter_sum
        '''
        
            Edge normalization trick
            - one_hot: (num_edge, num_relation)
            - deg: (num_node, num_relation)
            - index: (num_edge)
            - deg[edge_index[0]]: (num_edge, num_relation)
            - edge_norm: (num_edge)
        '''
        one_hot = tf.one_hot(tf.cast(edge_type, np.int32),
                             2 * num_relation, dtype=tf.int64)
        one_hot = tf.constant(one_hot.numpy())
        deg = scatter_sum(one_hot, edge_index[0], dim=0, dim_size=num_entity)
        index = edge_type + tf.keras.backend.arange(len(edge_index[0])) * (2 * num_relation)
        edge_norm = 1 / np.reshape(deg[edge_index[0]], -1)[index]

        return edge_norm

    def generate_sampled_graph_and_labels(self, triplets, batch_size, split_size, num_entity, num_rels, negative_rate):
        """
            Get training graph and signals
            First perform edge neighborhood sampling on graph, then perform negative
            sampling to generate negative samples
        """

        edges = self.sample_edge_uniform(len(triplets), batch_size)

        # Select sampled edges
        edges = triplets[edges]
        src, rel, dst = edges.transpose()
        uniq_entity, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # Negative sampling
        samples, labels = self.negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)
        # samples 是所有的三元组，labels是表示该三元组是真是假
        # further split graph, only half of the edges will be used as graph
        # structure, while the rest half is used as unseen positive samples
        split_size = int(batch_size * split_size)
        graph_split_ids = np.random.choice(np.arange(batch_size),
                                           size=split_size, replace=False)

        src = tf.constant(src[graph_split_ids], dtype=tf.float32)
        dst = tf.constant(dst[graph_split_ids], dtype=tf.float32)
        rel = tf.constant(rel[graph_split_ids], dtype=tf.float32)
        # Create bi-directional graph
        src, dst = tf.concat((src, dst), axis=0), tf.concat((dst, src), axis=0)
        rel = tf.concat((rel, rel + num_rels), axis=0)
        edge_type = rel
        self.edge_index = tf.stack((src, dst))
        self.entity = tf.constant(uniq_entity)
        self.edge_type = edge_type
        self.edge_norm = tf.ones(edge_type.shape)

        self.samples = tf.constant(samples)
        self.labels = tf.constant(labels)


class Planetoid:
    def __init__(self, name, data_dir="data", loop=True, norm=True):
        self.name = name
        self.loop = loop
        self.norm = norm
        self.data_dir = data_dir
        self.url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
        self.download()

    def download(self):
        output_dir = os.path.join(self.data_dir, self.name)
        for name in self.raw_file():
            file_name = "{}/{}".format(output_dir, name)
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            if not os.path.exists(file_name):
                url = "{}/{}".format(self.url, name)
                print('Downloading', url)
                data = urllib.request.urlopen(url)
                with open(file_name, 'wb') as wf:
                    wf.write(data.read())

    def raw_file(self):
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
        return ['ind.{}.{}'.format(self.name.lower(), name) for name in names]

    def parse_index_file(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def feature_normalize(self, features):
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def load(self):
        names = self.raw_file()
        objects = []
        for name in names[:-1]:
            f = open("data/{}/{}".format(self.name, name), 'rb')
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
            f.close()

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_reorder = self.parse_index_file("data/{}/{}".format(self.name, names[-1]))
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        # 调整顺序
        features[test_idx_reorder, :] = features[test_idx_range, :]

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj.tocoo().astype(np.float32)
        adj = tf.constant([[row, col] for row, col in zip(adj.row, adj.col)], dtype=tf.int32)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()  # 1000
        idx_train = range(len(y))  # 140
        idx_val = range(len(y), len(y) + 500)  # 500

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)

        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        features = np.array(features.todense(), dtype=np.float32)

        if self.norm:
            features = self.feature_normalize(features)
        if self.loop:
            adj = [add_remain_self_loop(adj, len(features))]
        else:
            adj = [adj]

        return features, adj, y_train, y_val, y_test, train_mask, val_mask, test_mask


class TUDataset:
    def __init__(self, name, split, data_dir="data"):
        self.name = name
        self.data_dir = data_dir
        self.split = split
        self.url = "http://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets"
        if not os.path.exists("{}/{}".format(data_dir, name)):
            self.download()
        self.x, self.y, self.edge_index, self.edge_attr, self.num_nodes, self.batch = self.read_data(self.data_dir,
                                                                                                     self.name)
        if self.split in [5, 10]:
            kflod = StratifiedKFold(split, shuffle=True)
            self.index_list = list(kflod.split(np.zeros(len(self.y)), self.y))

    def raw_file(self):
        names = ['A', "graph_indicator"]
        return ["{}_{}.txt".format(self.name, name) for name in names]

    def unzip(self, filename, folder):
        with zipfile.ZipFile(filename, 'r') as f:
            f.extractall(folder)

    def read_file(self, folder, prefix, name, dtype):
        path = os.path.join(folder, "{}/{}_{}.txt".format(prefix, prefix, name))
        return self.read_raw_text(path, seq=',', dtype=dtype)

    def read_raw_text(self, path, seq=None, start=0, end=None, dtype=None):
        with open(path, 'r') as rf:
            src = rf.read().split('\n')[:-1]
        src = [[dtype(x) for x in line.split(seq)[start:end]] for line in src]
        return np.array(src, dtype=dtype)

    def download(self):
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        url = "{}/{}.zip".format(self.url, self.name)
        outpath = "{}/{}.zip".format(self.data_dir, self.name)
        print('Downloading', url)
        data = urllib.request.urlopen(url)
        with open(outpath, 'wb') as wf:
            wf.write(data.read())
        self.unzip(outpath, self.data_dir)
        os.unlink(outpath)

    def read_data(self, folder, prefix):
        files = glob.glob(os.path.join(folder, '{}/{}_*.txt'.format(prefix, prefix)))
        names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]
        edge_index = self.read_file(folder, prefix, 'A', dtype=int) - 1  # 从0开始编码
        batch = self.read_file(folder, prefix, 'graph_indicator', dtype=int) - 1  # 从0开始编码
        node_attributes = node_labels = None
        if 'node_attributes' in names:
            node_attributes = self.read_file(folder, prefix, 'node_attributes', dtype=float)
        if 'node_labels' in names:
            node_labels = self.read_file(folder, prefix, 'node_labels', dtype=int)
            node_labels = node_labels - node_labels.min(0)[0]
            node_labels = np.reshape(node_labels, [-1])
            node_labels = np.eye(len(set(node_labels)))[node_labels]  # one_hot
        x = self.cat([node_attributes, node_labels])
        edge_attributes, edge_labels = None, None
        if 'edge_attributes' in names:
            edge_attributes = self.read_file(folder, prefix, 'edge_attributes', dtype=float)
        if 'edge_labels' in names:
            edge_labels = self.read_file(folder, prefix, 'edge_labels', dtype=int)
            edge_labels = edge_labels - edge_labels.min(0)[0]
            edge_labels = np.reshape(edge_labels, [-1])

            edge_labels = np.eye(len(set(edge_labels)))[edge_labels]
        edge_attr = self.cat([edge_attributes, edge_labels])
        y = None
        if 'graph_attributes' in names:  # Regression problem.
            y = self.read_file(folder, prefix, 'graph_attributes', dtype=float)
        elif 'graph_labels' in names:  # Classification problem.
            y = self.read_file(folder, prefix, 'graph_labels', dtype=int)
            _, _, y = np.unique(y, return_index=True, return_inverse=True)
            y = np.reshape(y, y.shape)
        num_nodes = edge_index.max() + 1 if x is None else len(node_labels)
        # edge_index, edge_attr = remove_self_loop(edge_index, edge_attr)
        # edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
        return x, y, edge_index, edge_attr, num_nodes, batch

    def cat(self, seq):
        seq = [item for item in seq if item is not None]
        seq = [np.expand_dims(item, -1) if len(item.shape) == 1 else item for item in seq]
        return np.concatenate(seq, axis=-1) if len(seq) > 0 else None

    def load(self, block_index=None):
        if self.split < 1:
            sample_train = int(len(self.y) * self.split)
            train_index = np.random.choice(np.arange(len(self.y)), size=sample_train, replace=False)
            # train_index = np.arange(10)
            test_index = np.delete(np.arange(len(self.y)), train_index)

        elif self.split in [5, 10]:
            train_index = self.index_list[block_index][0]
            test_index = self.index_list[block_index][1]
        else:
            raise ValueError("Current split not support")
        tudata = TuData(train_index, test_index)
        trainslices, testslices, zero_start_edge_index = tudata.get_slice(self.x, self.y, self.edge_index,
                                                                          self.edge_attr, self.batch)

        train_data = tudata.sample_data(self.x, self.y, zero_start_edge_index, self.edge_attr, self.batch, trainslices)
        test_data = tudata.sample_data(self.x, self.y, zero_start_edge_index, self.edge_attr, self.batch, testslices)
        return train_data, test_data

        # return train_data.shuffle(1000).window(batch_size), test_data.shuffle(1000).window(batch_size)

    def sample(self, data, batch_size, iterator_per_epoch=50, mode="Train"):

        x, y, edge_index, edge_attr, batch = data
        if mode == "train":
            nedge_attr = None
            for i in range(iterator_per_epoch):
                index = np.random.permutation(len(x))[:batch_size]
                nx = [x[i] for i in index]
                ny = [y[i] for i in index]
                nedge_index = [edge_index[i] for i in index]
                if edge_attr != None:
                    nedge_attr = [edge_attr[i] for i in index]
                nbatch = [batch[i] for i in index]

                yield nx, ny, nedge_index, nedge_attr, nbatch
        elif mode == "test":
            nedge_attr = None
            index_list = list(range(len(x)))
            for i in range(0, len(x), batch_size):
                index = index_list[i:i + batch_size]
                nx = [x[i] for i in index]
                ny = [y[i] for i in index]
                nedge_index = [edge_index[i] for i in index]
                if edge_attr != None:
                    nedge_attr = [edge_attr[i] for i in index]
                nbatch = [batch[i] for i in index]

                yield nx, ny, nedge_index, nedge_attr, nbatch


class TuData:
    def __init__(self, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index

    def split(self, x, y, edge_index, edge_attr, batch):

        # batch = np.reshape(batch, [-1])
        node_slice = np.cumsum(np.bincount(batch), axis=0)

        node_slice = np.concatenate([[0], node_slice])

        row = edge_index[:, 0]

        edge_slice = np.cumsum(np.bincount(batch[row]), axis=0)
        edge_slice = np.concatenate([[0], edge_slice])

        zero_start_edge_index = edge_index - np.expand_dims(node_slice[batch[row]], 1)

        edge_slice = np.expand_dims(edge_slice, -1)
        edge_slice = np.concatenate([edge_slice[:-1], edge_slice[1:]], 1)

        slices = {'edge_index': edge_slice}
        if x is not None:
            node_slice = np.expand_dims(node_slice, -1)
            slices['x'] = np.concatenate([node_slice[:-1], node_slice[1:]], 1)
        if edge_attr is not None:
            slices['edge_attr'] = edge_slice
        if y is not None:
            if y.shape[0] == batch.shape[0]:
                slices['y'] = node_slice
            else:
                slices['y'] = np.arange(0, batch[-1] + 2, dtype=np.long)
        return slices, zero_start_edge_index

    def get_slice(self, x, y, edge_index, edge_attr, batch):
        batch = np.reshape(batch, [-1])
        slices, edge_index_zero_start = self.split(x, y, edge_index, edge_attr, batch)
        train_slices = {}
        for key in slices.keys():
            train_slices[key] = slices[key][self.train_index]

        test_slices = {}
        for key in slices.keys():
            test_slices[key] = slices[key][self.test_index]

        return train_slices, test_slices, edge_index_zero_start

    def sample_data(self, x, y, edge_index, edge_attr, batch, sample_index):
        batch = np.reshape(batch, [-1])
        for key, value in sample_index.items():
            if key == "x":
                x = [x[start_end[0]:start_end[1]].tolist() for start_end in value]
                batch = [batch[start_end[0]:start_end[1]].tolist() for start_end in value]
            elif key == "y":
                y = y[value]
            elif key == "edge_index":
                edge_index = [edge_index[start_end[0]:start_end[1]].tolist() for start_end in value]
            elif key == "edge_attr":
                edge_attr = [edge_attr[start_end[0]:start_end[1]].tolist() for start_end in value]

        # return tf.data.Dataset.from_generator(self.generator(x, y, edge_index, edge_attr, batch),
        #                                       (tf.float32, tf.int32, tf.int32, tf.float32, tf.int32))
        return x, y, edge_index, edge_attr, batch

    def generator(self, x, y, edge_index, edge_attr, batch):

        def gen():
            if edge_attr == None:
                for i, x_i in enumerate(x):
                    yield x[i], y[i], edge_index[i], edge_attr, batch[i]
            else:
                for i, x_i in enumerate(x):
                    yield x[i], y[i], edge_index[i], edge_attr[i], batch[i]

        return gen


class Sminarog():
    def __init__(self, data="R8", data_dir="data", embedding="glove50"):
        self.data_dir = data_dir
        self.data = data
        self.embedding = embedding
        self.url = "https://www.cs.umb.edu/~smimarog/textmining/datasets/"
        self.download()
        if not os.path.exists("glove"):
            self.download_glove()
        else:
            print("Warning: Embedding dictionary has exists, the code will skip download {}. "
                  "if you want to use word2vec embedding, you could put them in the "
                  "embedding dictionary.".format(embedding))

        self.word2embedding, self.words2index = self.map_word_to_embedding()

    @property
    def vocab_size(self):
        return len(self.words2index)

    def raw_file(self, data):
        file_name = ["{}-train-all-terms.txt".format(data.lower()), "{}-test-all-terms.txt".format(data.lower())]
        return file_name

    def download(self):
        data_dir = os.path.join(self.data_dir, self.data)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for name in self.raw_file(self.data.lower()):
            url = "{}/{}".format(self.url, name)
            outpath = "{}/{}".format(data_dir, name)
            if not os.path.exists(outpath):
                print('Downloading', url)
                data = urllib.request.urlopen(url)
                with open(outpath, 'wb') as wf:
                    wf.write(data.read())

    def unzip(self, filename, folder):
        with zipfile.ZipFile(filename, 'r') as f:
            f.extractall(folder)

    def download_glove(self):
        # if not os.path.exists("embedding"):
        #     os.mkdir("embedding")
        url = "http://downloads.cs.stanford.edu/nlp/data/{}"
        if self.embedding[:5] == "glove":
            url = url.format("glove.6B.zip")
            outpath = "{}.zip".format("glove.6B.zip")
            print('Downloading Glove...', url)
            data = urllib.request.urlopen(url)
            with open(outpath, 'wb') as wf:
                wf.write(data.read())
            self.unzip(outpath, "glove")
            os.unlink(outpath)
        else:
            raise ValueError("Currently only support glove embedding!")

    # @property
    # def edge_size(self):
    #     return self.edge_size
    def map_edge2index(self, edge_type_num, k=5):
        new_edge_type_index = {}
        index_set = {0}
        for key, value in edge_type_num.items():
            if value > k:
                new_edge_type_index[key] = len(index_set)
                index_set.add(len(index_set))
            else:
                new_edge_type_index[key] = 0
        return new_edge_type_index

    def map_node2index(self, nodes):
        node2index = {}
        for i, item in enumerate(nodes):
            node2index[item] = i
        return node2index

    def build_graph(self, edge2index=None, node2index=None, mode="train", p=2, k=5):
        names = self.raw_file(self.data)
        if mode == "train":
            name = names[0]
        elif mode == "test":
            name = names[1]
        else:
            raise ValueError("mode can only equal train or test")
        filename = os.path.join(os.path.join(self.data_dir, self.data), name)

        labels = []
        features = []
        adjs = []
        batchs = []
        edge_attrs = []
        rf = open(filename)
        edge_type_num = defaultdict(int)
        # edge_type_index = {}
        if node2index == None:
            node2index = {'<UNK>': 0}
        else:
            node2index = node2index

        graph_edge_map = []
        nodes_list = []
        for line in rf:
            word2index = {}
            adj = []
            line = line.strip().split('\t')
            label, text = line[0], line[1]
            text_list = text.split(' ')
            for w in set(text_list):
                word2index[w] = len(word2index)
            index2word = {v: k for k, v in word2index.items()}
            graph_edge_map.append(index2word)
            for i, source in enumerate(text_list):
                if source not in node2index and mode == "train":
                    node2index[source] = len(node2index)
                if i >= p:
                    targets = text_list[-p + i:p + i + 1]
                elif i < p:
                    targets = text_list[:p + i + 1]
                for target in targets:
                    adj.append([word2index[source], word2index[target]])
                    edge_type_num[(source, target)] += 1
                    # edge_type_index[(source, target)] = len(edge_type_index) + 1
                    # edge_attr.append([self.words2index.get(source, 0), self.words2index.get(target, 0)])
            # feature = [self.word2embedding.get(self.words2index.get(index2word[i], 0)) for i in range(len(word2index))]
            # feature = [index2word[i] for i in range(len(word2index))]
            # if return_node2index:
            node_per_text = [index2word[i] for i in range(len(word2index))]
            nodes_list.append(node_per_text)

            # features.append(feature)
            adjs.append(adj)
            labels.append(label)
            batchs.append([len(batchs)] * len(node_per_text))

            # edge_attrs.append(edge_attr)

        if edge2index == None:
            edge2index = self.map_edge2index(edge_type_num, k)
        else:
            edge2index = edge2index
        for i, adj in enumerate(adjs):
            edge_map = graph_edge_map[i]
            edge_attr = []
            for edge_pair in adj:
                source_w = edge_map[edge_pair[0]]
                target_w = edge_map[edge_pair[1]]

                edge_attr.append(edge2index.get((source_w, target_w), 0))
            edge_attrs.append(edge_attr)

        nodes = []
        for node_list in nodes_list:
            node_index = [node2index.get(w, 0) for w in node_list]
            nodes.append(node_index)

        # edge_size = len([key for key, value in edge_type_num.items() if value > k]) + 1
        print("Num of the class in {} is {}.".format(mode, len(set(labels))))
        label2index = {label: index for index, label in enumerate(list(set(labels)))}
        labels = [label2index[label] for label in labels]

        return nodes, adjs, edge_attrs, labels, batchs, edge2index, node2index

    @property
    def vocabs(self):
        names = self.raw_file(self.data)
        vocabs = defaultdict(int)
        filename = os.path.join(os.path.join(self.data_dir, self.data), names[0])
        rf = open(filename)
        for line in rf:
            line = line.strip().split('\t')
            vocab_set = [i for i in line[1].split(' ')]
            vocab_num = Counter(vocab_set)
            for vocab, num in vocab_num.items():
                vocabs[vocab] += num
        return vocabs

    def map_word_to_embedding(self):
        unknown = np.random.uniform(-1, 1, size=(int(self.embedding[5:]),)).astype(np.float32).tolist()
        word2embedding = {"<UNK>": unknown}
        words2index = {"<UNK>": 0}
        if self.embedding == "glove50":
            glove_path = "{}/glove.6B.50d.txt".format(self.embedding[:5])
        elif self.embedding == "glove100":
            glove_path = "{}/glove.6B.100d.txt".format(self.embedding[:5])
        elif self.embedding == "glove200":
            glove_path = "{}/glove.6B.200d.txt".format(self.embedding[:5])
        elif self.embedding == "glove300":
            glove_path = "{}/glove.6B.300d.txt".format(self.embedding[:5])
        else:
            raise ValueError("glove_path can only in glove50, glove100, glove200, glove300 !")

        with codecs.open(glove_path, encoding='utf-8') as rf:
            # vocabs = self.vocabs
            for line in rf:
                line = line.strip().split(' ')
                word = line[0]
                # if word in vocabs and vocabs[word]>=5:
                embedding = [float(i) for i in line[1:]]
                word2embedding[float(len(words2index))] = embedding  # 这里是为了后面能够使用
                word2embedding[word] = embedding  # 这里是为了后面能够使用
                words2index[word] = len(words2index)

        return word2embedding, words2index

    def generator(self, nodes, adjs, edge_attrs, labels, batchs):
        def gen():
            if edge_attrs == None:
                for i, feature_i in enumerate(nodes):
                    yield nodes[i], labels[i], adjs[i], edge_attrs, batchs[i]
            else:
                for i, feature_i in enumerate(nodes):
                    yield nodes[i], labels[i], adjs[i], edge_attrs[i], batchs[i]

        return gen

    def load(self, nodes, adjs, labels, edge_attrs=None, batchs=None, batch_size=32):
        data = tf.data.Dataset.from_generator(self.generator(nodes, adjs, edge_attrs, labels, batchs),
                                              (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))
        return data.shuffle(1000).prefetch(tf.data.experimental.AUTOTUNE).window(batch_size)

    def load_graph(self, mode="train"):
        nodes = np.load("data/R8/{}_nodes.npy".format(mode), allow_pickle=True)
        adjs = np.load("data/R8/{}_edge_lists.npy".format(mode), allow_pickle=True)
        edge_attrs = np.load("data/R8/{}_edge_weights.npy".format(mode), allow_pickle=True)
        labels = np.load("data/R8/{}_labels.npy".format(mode), allow_pickle=True)
        batchs = np.load("data/R8/{}_batchs.npy".format(mode), allow_pickle=True)
        node2index = np.load("data/R8/node2index.npy", allow_pickle=True).item()
        return nodes, adjs, edge_attrs, labels, batchs, node2index
