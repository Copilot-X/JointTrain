import tensorflow as tf
import scipy.sparse as sp
import numpy as np

class GCN(object):

    def __init__(self, is_training, drop_prob, num_rela, dims):
        self.is_training = is_training
        self.dropout = drop_prob
        self.num_rela = num_rela
        self.dims = dims
        self.supports = None
        self.num_features_nonzero = None

    def __preprocess_features__(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return self.__sparse_to_tuple__(sp.csr_matrix(features))

    def __adj_normalized__(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def __sparse_to_tuple__(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return (coords, values, shape)

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def __preprocess_adj__(self, adj):
        # Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
        adj_normalized = self.__adj_normalized__(adj + sp.eye(adj.shape[0]))
        return self.__sparse_to_tuple__(adj_normalized)

    def preprocess(self, features, adjs):
        # preprocess features
        """
        features: [N nodes, feature]
        adjs: [h2r_adj, r2t_adj, self_adj], each adj is csr_matrix with shape: [N nodes, N nodes]
        """
        features = self.__preprocess_features__(features)
        supports = list()
        for i in range(len(adjs)):
            supports.append(self.__preprocess_adj__(adjs[i]))
        return features, supports

    def __dot__(self, x, y, sparse=False):
        """Wrapper for tf.matmul (sparse vs dense)."""
        if sparse:
            res = tf.sparse_tensor_dense_matmul(x, y)
        else:
            res = tf.matmul(x, y)
        return res

    def __glorot__(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0/(shape[0]+shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def __sparse_dropout__(self, x, keep_prob, noise_shape):
        """Dropout for sparse tensors."""
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(x, dropout_mask)
        return pre_out * (1./keep_prob)

    def __gcnLayer__(self, layer_id, input_dim, output_dim, inputs,
                     sparse_inputs=False, act=tf.nn.relu, bias=False):

        # weights and bias
        vars = dict()
        with tf.variable_scope('gcn_layer_' + str(layer_id)):
            for i in range(3):
                w_id = str(layer_id) + str(i)
                vars['weights_'+w_id] = self.__glorot__([input_dim, output_dim], name='weights_'+w_id)
                if bias:
                    initial = tf.zeros([output_dim],  dtype=tf.float32)
                    vars['bias'+w_id] = tf.Variable(initial, name='bias_'+w_id)

        # drop out
        if self.is_training:
            if sparse_inputs:
                for adj_idx in range(3):
                    inputs[adj_idx] = self.__sparse_dropout__(inputs[adj_idx], 1-self.dropout, self.num_features_nonzero)
            else:
                for i in range(3):
                    inputs[i] = tf.nn.dropout(inputs[i], 1-self.dropout)

        # convolve
        outputs = []
        for adj_idx in range(3):
            out = self.__dot__(inputs[adj_idx], vars['weights_'+w_id], sparse=sparse_inputs)
            out = self.__dot__(self.supports[adj_idx], out, sparse=True)
            if bias:
                outs = outs + vars['bias_'+w_id]
            outputs.append(act(out))

        # summary
        for key in vars:
            tf.summary.histogram(key, vars[key])

        return outputs

    def __merge__(self, inputs, act=lambda x: x):
        return act(tf.add_n(inputs))

    def gcn(self, features, supports, num_features_nonzero):
        self.supports = supports
        self.num_features_nonzero = num_features_nonzero
        layer_num = len(self.dims) - 1
        # layer builder
        outputs = self.__gcnLayer__(layer_id=0,
                                    input_dim=self.dims[0],
                                    output_dim=self.dims[1],
                                    inputs=[features for _ in range(3)],
                                    sparse_inputs=True,
                                    act=tf.nn.relu)
        for i in range(1, layer_num):
            outputs = self.__gcnLayer__(layer_id=i,
                                        input_dim=self.dims[i],
                                        output_dim=self.dims[i+1],
                                        inputs=outputs,
                                        act=tf.nn.relu)
        outputs = self.__merge__(outputs)
        return outputs[:self.num_rela]

