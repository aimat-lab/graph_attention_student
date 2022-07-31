import sys
from typing import List

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.modules import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate
from kgcnn.layers.modules import DropoutEmbedding, DenseEmbedding
from kgcnn.layers.conv.attention import AttentionHeadGATV2


class MultiChannelGatLayer(AttentionHeadGATV2):
    """
    This class is mostly a copy and modification of ``kgcnn.layers.conv.attention.AttentionHeadGATV2``.
    """
    def __init__(self,
                 *args,
                 units: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 channels: int = 1,
                 attention_dropout_rate: float = 0,
                 **kwargs):
        super(MultiChannelGatLayer, self).__init__(
            *args,
            units=units,
            activation=activation,
            use_bias=use_bias,
            **kwargs
        )

        self.attention_dropout_rate = attention_dropout_rate
        self.channels = channels
        self.lay_dropout = DropoutEmbedding(rate=attention_dropout_rate)

        self.alpha_layers = []
        for i in range(channels):
            lay_alpha_activation = DenseEmbedding(units, activation=activation, use_bias=use_bias)
            lay_alpha = DenseEmbedding(1, activation="linear", use_bias=False)
            self.alpha_layers.append((lay_alpha_activation, lay_alpha))

        self.lay_concat_alphas = LazyConcatenate(axis=-2)
        self.lay_concat_embeddings = LazyConcatenate(axis=-2)

    def call(self, inputs, **kwargs):
        """Forward pass.

        N: Number of nodes in the graph
        V: Number of embeddings per node for the previous layer.
        K: Number of channels

        M: Number of edges in the graph
        F: Number of edge embeddings

        Args:
            inputs (list): of [nodes, edges, edge_indices]
                - nodes (tf.RaggedTensor): Node embeddings of shape ([batch], [N], K, V)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape ([batch], [M], F)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape ([batch], [M], 2)
        Returns:
            tf.RaggedTensor: Embedding tensor of pooled edge attentions for each node.
        """
        nodes, edge, edge_index = inputs

        # "a_ij" is a single-channel edge attention coefficient tensor. "h_i" is a single-channel node
        # embedding vector. These lists collect the single tensors for each channel and at the end are
        # concat into the multi-channel tensors.
        a_ijs = []
        h_is = []
        for k, (lay_alpha_activation, lay_alpha) in enumerate(self.alpha_layers):

            node = nodes[:, :, k, :]

            # Copied from the original class
            w_n = self.lay_linear_trafo(node, **kwargs)
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)
            wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
            if self.use_edge_features:
                e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
            else:
                e_ij = self.lay_concat([n_in, n_out], **kwargs)

            a_ij = lay_alpha_activation(e_ij, **kwargs)
            a_ij = lay_alpha(a_ij, **kwargs)

            # Added to hopefully improve overfit problem
            a_ij = self.lay_dropout(a_ij)

            h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)

            if self.use_final_activation:
                h_i = self.lay_final_activ(h_i, **kwargs)

            a_ij = tf.expand_dims(a_ij, axis=-2)
            a_ijs.append(a_ij)

            h_i = tf.expand_dims(h_i, axis=-2)
            h_is.append(h_i)

        a_ijs = self.lay_concat_alphas(a_ijs)
        h_is = self.lay_concat_embeddings(h_is)

        # An important modification we need here is that this layer also returns the attention coefficients
        # because we calculate the edge importances from those.
        # h_is: ([batch], [N], K, V)
        # a_ijs: ([batch], [M], K, 1)
        return h_is, a_ijs


class ExplanationSparsityRegularization(GraphBaseLayer):

    def __init__(self, coef: float = 1.0, **kwargs):
        super(ExplanationSparsityRegularization, self).__init__(**kwargs)
        self.coef = coef

    def call(self, inputs):
        # importances: ([batch], [N], K)
        importances = inputs

        loss = tf.reduce_sum(importances, axis=-1)
        loss = tf.reduce_mean(loss)
        self.add_loss(self.coef * loss)

        return


class ExplanationExclusivityRegularization(GraphBaseLayer):

    def __init__(self, coef: float = 1.0, **kwargs):
        super(ExplanationExclusivityRegularization, self).__init__(**kwargs)
        self.coef = coef

    def call(self, inputs):
        # importances: ([batch], [N], K)
        importances = inputs

        # This loss function encourages the sum of all the elements along the importance axis to be =1.
        # This means that every element needs to be explained in some way.
        # It also discourages this sum of 1 to be reached by various small values and rather encourages
        # one item being =1 while the others are 0 - aka an explanation to be exclusive to one channel.
        square_part = tf.square(tf.reduce_sum(tf.sqrt(importances + 0.1), axis=-1) - 1)
        sqrt_part = - tf.sqrt(tf.reduce_sum(importances, axis=-1) + 0.1)
        loss = square_part - sqrt_part

        loss = tf.reduce_mean(loss)
        self.add_loss(self.coef * loss)


class StaticMultiplicationEmbedding(GraphBaseLayer):

    def __init__(self,
                 values: List[float],
                 bias: float = 0.0,
                 **kwargs):
        GraphBaseLayer.__init__(self)
        self.values = tf.convert_to_tensor(values, dtype=tf.float32)
        self.bias = bias

    def call(self, inputs):
        loss = tf.reduce_sum(inputs, axis=-1)
        loss = tf.reduce_mean(loss)
        #self.add_loss(0.1 * loss)
        return tf.reduce_sum(inputs * self.values, axis=-1) + self.bias
