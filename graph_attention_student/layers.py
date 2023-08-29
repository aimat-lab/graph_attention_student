import sys
import typing as t
from typing import List

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.partition import partition_row_indexing
from kgcnn.ops.segment import segment_ops_by_name, segment_softmax
from kgcnn.layers.modules import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate, LazyAverage, LazyAdd
from kgcnn.layers.modules import DropoutEmbedding, DenseEmbedding, ActivationEmbedding
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.conv.gat_conv import PoolingLocalEdgesAttention
from kgcnn.layers.conv.gat_conv import AttentionHeadGATV2


class CoefficientActivation(GraphBaseLayer):

    def __init__(self,
                 activation: str = 'relu',
                 coefficient: float = 1.0,
                 ):
        super(CoefficientActivation, self).__init__()
        self.activation = activation
        self.coefficient = coefficient

        self.lay_activation = ActivationEmbedding(activation=activation)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        return self.coefficient * self.lay_activation(x)


class ExtendedPoolingLocalEdgesAttention(PoolingLocalEdgesAttention):
    r"""
    """

    def __init__(self, pooling_index=0, **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesAttention, self).__init__(**kwargs)
        self.pooling_index = pooling_index

    def call(self, inputs, **kwargs):
        """Forward pass.
        Args:
            inputs: [node, edges, attention, edge_indices]
                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - attention (tf.RaggedTensor): Attention coefficients of shape (batch, [M], 1)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], F)
        Returns:
            tf.RaggedTensor: Embedding tensor of pooled edge attentions for each node of shape (batch, [N], F)
        """
        # Need ragged input but can be generalized in the future.
        self.assert_ragged_input_rank(inputs)
        # We cast to values here
        nod, node_part = inputs[0].values, inputs[0].row_lengths()
        edge = inputs[1].values
        attention = inputs[2].values
        edgeind, edge_part = inputs[3].values, inputs[3].row_lengths()

        shiftind = partition_row_indexing(edgeind, node_part, edge_part, partition_type_target="row_length",
                                          partition_type_index="row_length", to_indexing='batch',
                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, self.pooling_index]  # Pick first index eg. ingoing
        dens = edge
        ats = attention
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            ats = tf.gather(ats, node_order, axis=0)

        # Apply segmented softmax
        ats = segment_softmax(ats, nodind)
        get = dens * ats
        get = tf.math.segment_sum(get, nodind)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            get = tf.scatter_nd(tf.keras.backend.expand_dims(tf.range(tf.shape(get)[0]), axis=-1), get,
                                tf.concat([tf.shape(nod)[:1], tf.shape(get)[1:]], axis=0))

        out = tf.RaggedTensor.from_row_lengths(get, node_part, validate=self.ragged_validate)
        ats = tf.RaggedTensor.from_row_lengths(ats, node_part, validate=self.ragged_validate)

        return out, ats

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesAttention, self).get_config()
        config.update({"pooling_index": self.pooling_index})
        return config


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
                 share_weights: bool = False,
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

        if share_weights:
            lay_linear = DenseEmbedding(units, activation='linear', use_bias=use_bias)

        self.alpha_layers = []
        for i in range(channels):
            if not share_weights:
                lay_linear = DenseEmbedding(units, activation='linear', use_bias=use_bias)

            lay_alpha_activation = DenseEmbedding(units, activation=activation, use_bias=use_bias)
            lay_alpha = DenseEmbedding(1, activation="linear", use_bias=False)
            self.alpha_layers.append((lay_linear, lay_alpha_activation, lay_alpha))

        self.lay_concat_alphas = LazyConcatenate(axis=-2)
        self.lay_concat_embeddings = LazyConcatenate(axis=-2)
        self.lay_pool_attention = ExtendedPoolingLocalEdgesAttention()

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
        for k, (lay_linear, lay_alpha_activation, lay_alpha) in enumerate(self.alpha_layers):

            node = nodes[:, :, k, :]

            # Copied from the original class
            w_n = lay_linear(node, **kwargs)
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

            h_i, ats_ij = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
            #print(a_ij.shape, ats_ij.shape)

            if self.use_final_activation:
                h_i = self.lay_final_activ(h_i, **kwargs)

            #a_ij = ks.activations.softmax(a_ij, axis=-2)
            a_ij = tf.expand_dims(a_ij, axis=-2)
            a_ijs.append(a_ij)
            # ats_ij = tf.expand_dims(ats_ij, axis=-2)
            # a_ijs.append(ats_ij)

            h_i = tf.expand_dims(h_i, axis=-2)
            h_is.append(h_i)

        a_ijs = self.lay_concat_alphas(a_ijs)
        h_is = self.lay_concat_embeddings(h_is)

        # An important modification we need here is that this layer also returns the attention coefficients
        # because we calculate the edge importances from those.
        # h_is: ([batch], [N], K, V)
        # a_ijs: ([batch], [M], K, 1)
        return h_is, a_ijs


class MultiHeadGatLayer(AttentionHeadGATV2):
    """
    This class is mostly a copy and modification of ``kgcnn.layers.conv.attention.AttentionHeadGATV2``.
    """
    def __init__(self,
                 *args,
                 units: int,
                 num_heads: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 concat_heads: bool = False,
                 attention_dropout_rate: float = 0,
                 share_weights: bool = False,
                 **kwargs):
        super(MultiHeadGatLayer, self).__init__(
            *args,
            units=units,
            activation=activation,
            use_bias=use_bias,
            **kwargs
        )

        self.attention_dropout_rate = attention_dropout_rate
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.lay_dropout = DropoutEmbedding(rate=attention_dropout_rate)

        if share_weights:
            lay_linear = DenseEmbedding(units, activation='linear', use_bias=use_bias)

        self.head_layers: t.List[t.Tuple[DenseEmbedding, DenseEmbedding, DenseEmbedding]] = []
        for _ in range(num_heads):
            if not share_weights:
                lay_linear = DenseEmbedding(units, activation='linear', use_bias=use_bias)

            lay_alpha_activation = DenseEmbedding(units, activation=activation, use_bias=use_bias)
            lay_alpha = DenseEmbedding(1, activation="linear", use_bias=False)

            # Adding all the layers as a tuple to the list as representing one of the heads
            self.head_layers.append((lay_linear, lay_alpha_activation, lay_alpha))

        self.lay_concat_alphas = LazyConcatenate(axis=-2)
        self.lay_concat_embeddings = LazyConcatenate(axis=-2)
        self.lay_pool_attention = PoolingLocalEdgesAttention()
        self.lay_pool = PoolingLocalEdges()

        self.lay_average_heads = LazyAverage()
        self.lay_concat_heads = LazyConcatenate(axis=-1)

    def call(self, inputs, **kwargs):
        """Forward pass.

        N: Number of nodes in the graph
        V: Number of embeddings per node for the previous layer.
        Vu: Number of embeddings per node after the this layer
        K: Number of attention heads

        M: Number of edges in the graph
        F: Number of edge embeddings

        Brackets [] indicate a ragged dimension

        Args:
            inputs (list): of [nodes, edges, edge_indices]
                - nodes (tf.RaggedTensor): Node embeddings of shape ([batch], [N], V)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape ([batch], [M], F)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape ([batch], [M], 2)
        Returns:
            Tuple consisting of:
                - nodes (tf.RaggedTensor): Node embeddings of shape ([batch], [N], Vu * K) if head concat
                  is used and ([batch], [N], Vu) if it is not used and heads are averaged instead
                - attention coefficients (tf.RaggedTensor): The attention coefficients for the graph edges
                  of the various attention heads of shape ([batch], [M], K)
        """
        node, edge, edge_index = inputs

        # "a_ij" is a single-channel edge attention coefficient tensor. "h_i" is a single-channel node
        # embedding vector. These lists collect the single tensors for each channel and at the end are
        # concat into the multi-channel tensors.
        a_ijs = []
        h_is = []
        for k, (lay_linear, lay_alpha_activation, lay_alpha) in enumerate(self.head_layers):

            # Copied from the original class
            w_n = lay_linear(node, **kwargs)
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)
            wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
            if self.use_edge_features:
                e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
            else:
                e_ij = self.lay_concat([n_in, n_out], **kwargs)

            # a_ij: ([batch], [M], 1)
            a_ij = lay_alpha_activation(e_ij, **kwargs)
            a_ij = lay_alpha(a_ij, **kwargs)

            # Added to hopefully improve overfit problem
            a_ij = self.lay_dropout(a_ij)

            # This is my own attempt at implementing an attention pooling operation. The problem with the
            # kgcnn PoolingLocalEdgesAttention is that it uses attention values which are not [0, 1] range
            # and as far as I can see there is also no way to get these actual coefficients out of the
            # process either.
            # I think this should work, because I realized that softmax should work on a ragged dimension as
            # well (at least it did not cause errors till now).
            # a_ij = ks.activations.softmax(a_ij, axis=-2)
            # h_i = self.lay_pool([node, wn_out * a_ij, edge_index], **kwargs)

            h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
            #a_ij = ks.activations.softmax(a_ij, axis=-2)

            if self.use_final_activation:
                h_i = self.lay_final_activ(h_i, **kwargs)

            a_ij = tf.expand_dims(a_ij, axis=-2)
            a_ijs.append(a_ij)

            #h_i = tf.expand_dims(h_i, axis=-2)
            h_is.append(h_i)

        a_ijs = self.lay_concat_alphas(a_ijs)

        if self.concat_heads:
            h_is = self.lay_concat_heads(h_is)
        else:
            h_is = self.lay_average_heads(h_is)

        # An important modification we need here is that this layer also returns the attention coefficients
        # because we calculate the edge importances from those.
        # h_is: ([batch], [N], K * Vu) or ([batch], [N], Vu)
        # a_ijs: ([batch], [M], K, 1)
        return h_is, a_ijs


class ExplanationSparsityRegularization(GraphBaseLayer):

    def __init__(self,
                 coef: float = 1.0,
                 factor: t.Optional[float] = None,
                 **kwargs):
        super(ExplanationSparsityRegularization, self).__init__(**kwargs)
        self.factor = coef
        if factor is not None:
            self.factor = factor

    def call(self, inputs):
        # importances: ([batch], [N], K)
        importances = inputs

        loss = tf.reduce_mean(tf.math.abs(importances))
        self.add_loss(loss * self.factor)


class ExplanationGiniRegularization(GraphBaseLayer):

    def __init__(self,
                 factor: float,
                 num_channels: int,
                 **kwargs):
        super(ExplanationGiniRegularization, self).__init__(**kwargs)
        self.factor = factor
        self.num_channels = num_channels

    def call(self, inputs, *args, **kwargs):
        # importances: ([batch], [N], K)
        importances = inputs
        importances_reduced = tf.reduce_mean(importances, axis=-2)

        values = []
        values_reduced = []
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                diff = tf.abs(importances[:, :, i] - importances[:, :, j])
                diff = tf.expand_dims(diff, axis=-1)
                values.append(diff)

                diff_reduced = tf.abs(importances_reduced[:, i] - importances_reduced[:, j])
                diff_reduced = tf.expand_dims(diff_reduced, axis=-1)
                values_reduced.append(diff_reduced)

        values = tf.concat(values, axis=-1)
        loss = tf.reduce_sum(values, axis=-1)
        loss /= ((2 * self.num_channels**2 - self.num_channels) * tf.reduce_mean(importances, axis=-1))
        self.add_loss(-self.factor * tf.reduce_mean(loss))
        #self.add_loss(self.factor / tf.reduce_mean(loss))

        values_reduced = tf.concat(values_reduced, axis=-1)
        loss_reduced = tf.reduce_sum(values_reduced, axis=-1)
        loss_reduced /= ((2 * self.num_channels**2 - self.num_channels) * tf.reduce_mean(importances_reduced, axis=-1))
        self.add_loss(2 * self.factor * tf.reduce_mean(loss_reduced))


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
        #square_part = tf.square(tf.reduce_sum(tf.sqrt(importances + 0.1), axis=-1) - 1)
        #sqrt_part = - tf.sqrt(tf.reduce_sum(importances, axis=-1) + 0.1)
        #loss = square_part - sqrt_part

        mean = tf.reduce_mean(importances, axis=-1, keepdims=True)
        mean = tf.concat([mean for _ in range(importances.shape[-1])], axis=-1)
        loss = tf.reduce_sum(tf.square(importances - mean), axis=-1)
        loss = tf.reduce_mean(loss)
        self.add_loss(-self.coef * loss)


class StaticMultiplicationEmbedding(GraphBaseLayer):

    def __init__(self,
                 values: List[float],
                 bias: float = 0.0,
                 **kwargs):
        GraphBaseLayer.__init__(self)
        self.values = tf.convert_to_tensor(values, dtype=tf.float32)
        self.bias = bias

    def call(self, inputs):
        return tf.reduce_sum(inputs * self.values, axis=-1) + self.bias


class MultiHeadGATV2Layer(AttentionHeadGATV2):

    def __init__(self,
                 units: int,
                 num_heads: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 concat_heads: bool = True,
                 concat_self: bool = False,
                 **kwargs):
        super(MultiHeadGATV2Layer, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            **kwargs
        )
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.concat_self = concat_self

        self.head_layers = []
        for _ in range(num_heads):
            lay_activation = DenseEmbedding(units, activation='relu', use_bias=use_bias)
            lay_linear = DenseEmbedding(units, activation='linear', use_bias=use_bias)
            lay_alpha_activation = DenseEmbedding(units, activation=activation, use_bias=use_bias)
            lay_alpha = DenseEmbedding(1, activation='linear', use_bias=False)

            # self.head_layers.append((lay_linear, lay_alpha_activation, lay_alpha))
            self.head_layers.append({
                'lay_activation': lay_activation,
                'lay_linear': lay_linear,
                'lay_alpha_activation': lay_alpha_activation,
                'lay_alpha': lay_alpha,
            })

        self.lay_concat_alphas = LazyConcatenate(axis=-2)
        self.lay_concat_embeddings = LazyConcatenate(axis=-2)
        self.lay_pool_attention = PoolingLocalEdgesAttention()

        if self.concat_heads:
            self.lay_combine_heads = LazyConcatenate(axis=-1)
        else:
            self.lay_combine_heads = LazyAverage()

    def __call__(self, 
                 inputs, 
                 edge_mask: t.Optional[tf.RaggedTensor] = None, 
                 **kwargs):
        node, edge, edge_index = inputs

        # "a_ij" is a single-channel edge attention logits tensor. "a_ijs" is consequently the list which
        # stores these tensors for each attention head.
        # "h_i" is a single-channel node embedding tensor. "h_is" is consequently the list which stores
        # these tensors for each attention head.
        a_ijs = []
        h_is = []
        for k, layers in enumerate(self.head_layers):

            # These are essentially the message embeddings that we broadcast during the message 
            # passing step to then compute the new node embeddings. These are derived through a very shallow MLP 
            # from the current node embeddings
            # w_n: ([B], [V], N)
            w_n = node
            # w_n = layers['lay_activation'](w_n, **kwargs)
            w_n = layers['lay_linear'](w_n, **kwargs)
            
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)
            wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
            if self.use_edge_features:
                e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
            else:
                e_ij = self.lay_concat([n_in, n_out], **kwargs)

            # a_ij: ([B], [E], 1)
            a_ij = layers['lay_alpha_activation'](e_ij, **kwargs)
            a_ij = layers['lay_alpha'](a_ij, **kwargs)
            
            # Edge mask is supposed to be a binary mask (only 1 and 0) whose primary usage is to completely mask out certain edges
            # during the message passing operation by setting the corresponding attention weight to zero.
            if edge_mask is not None:
                a_ij *= edge_mask

            # h_i: ([B], [V], N)
            h_i = self.lay_pool_attention([w_n, wn_out, a_ij, edge_index], **kwargs)
            
            if self.concat_self:
                h_i = tf.concat([h_i, w_n], axis=-1)
            
            if self.use_final_activation:
                h_i = self.lay_final_activ(h_i, **kwargs)

            # a_ij after expand: ([B], [E], 1, 1)
            a_ij = tf.expand_dims(a_ij, axis=-2)
            a_ijs.append(a_ij)

            # h_i = tf.expand_dims(h_i, axis=-2)
            h_is.append(h_i)

        a_ijs = self.lay_concat_alphas(a_ijs)

        h_is = self.lay_combine_heads(h_is)

        # An important modification we need here is that this layer also returns the attention coefficients
        # because in MEGAN we need those to calculate the edge attention values with!
        # h_is: ([batch], [N], K * Vu) or ([batch], [N], Vu)
        # a_ijs: ([batch], [M], K, 1)
        return h_is, a_ijs

    def get_config(self):
        """Update layer config."""
        config = super(MultiHeadGATV2Layer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'concat_heads': self.concat_heads
        })

        return config



class MultiHeadAttentionLayer(GraphBaseLayer):

    def __init__(self,
                 units: int,
                 num_heads: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 concat_heads: bool = True
                 ):
        
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads

        self.head_layers = []
        for _ in range(num_heads):
            lay_linear = DenseEmbedding(units, activation='linear', use_bias=use_bias)
            lay_alpha_activation = DenseEmbedding(units, activation=activation, use_bias=use_bias)
            lay_alpha = DenseEmbedding(1, activation='linear', use_bias=False)

            self.head_layers.append((lay_linear, lay_alpha_activation, lay_alpha))

        self.lay_concat_alphas = LazyConcatenate(axis=-2)
        self.lay_concat_embeddings = LazyConcatenate(axis=-2)
        self.lay_pool_attention = PoolingLocalEdgesAttention()

        if self.concat_heads:
            self.lay_combine_heads = LazyConcatenate(axis=-1)
        else:
            self.lay_combine_heads = LazyAverage()

    def __call__(self, 
                 inputs, 
                 edge_mask: t.Optional[tf.RaggedTensor] = None, 
                 **kwargs):
        node, edge, edge_index = inputs

        # "a_ij" is a single-channel edge attention logits tensor. "a_ijs" is consequently the list which
        # stores these tensors for each attention head.
        # "h_i" is a single-channel node embedding tensor. "h_is" is consequently the list which stores
        # these tensors for each attention head.
        a_ijs = []
        h_is = []
        for k, (lay_linear, lay_alpha_activation, lay_alpha) in enumerate(self.head_layers):

            # Copied from the original class
            w_n = lay_linear(node, **kwargs)
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)
            wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
            if self.use_edge_features:
                e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
            else:
                e_ij = self.lay_concat([n_in, n_out], **kwargs)

            # a_ij: ([batch], [M], 1)
            a_ij = lay_alpha_activation(e_ij, **kwargs)
            a_ij = lay_alpha(a_ij, **kwargs)
            
            # Edge mask is supposed to be a binary mask (only 1 and 0) whose primary usage is to completely mask out certain edges
            # during the message passing operation by setting the corresponding attention weight to zero.
            if edge_mask is not None:
                a_ij *= edge_mask

            # h_i: ([batch], [N], F)
            h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
            
            #h_i += w_n

            if self.use_final_activation:
                h_i = self.lay_final_activ(h_i, **kwargs)

            # a_ij after expand: ([batch], [M], 1, 1)
            a_ij = tf.expand_dims(a_ij, axis=-2)
            a_ijs.append(a_ij)

            # h_i = tf.expand_dims(h_i, axis=-2)
            h_is.append(h_i)

        a_ijs = self.lay_concat_alphas(a_ijs)

        h_is = self.lay_combine_heads(h_is)

        # An important modification we need here is that this layer also returns the attention coefficients
        # because in MEGAN we need those to calculate the edge attention values with!
        # h_is: ([batch], [N], K * Vu) or ([batch], [N], Vu)
        # a_ijs: ([batch], [M], K, 1)
        return h_is, a_ijs

    def get_config(self):
        """Update layer config."""
        config = super(MultiHeadGATV2Layer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'concat_heads': self.concat_heads
        })

        return config