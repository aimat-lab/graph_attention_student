from typing import List, Optional, Callable

import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.modules import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate, LazyAverage
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.pooling import PoolingWeightedNodes

from graph_attention_student.layers import MultiChannelGatLayer
from graph_attention_student.layers import ExplanationSparsityRegularization
from graph_attention_student.layers import ExplanationExclusivityRegularization


class MultiAttentionStudent(ks.models.Model):

    def __init__(self,
                 # primary network related arguments
                 units: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 # importance related arguments
                 use_edge_features: bool = True,
                 importance_channels: int = 2,
                 importance_activation: str = 'sigmoid',
                 importance_dropout_rate: float = 0.0,
                 final_units: List[int] = [],
                 final_activation: str = 'relu',
                 sparsity_factor: float = 0.0,
                 exclusivity_factor: float = 0.0,
                 lay_additional_cb: Optional[Callable] = None,
                 ):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_edge_features = use_edge_features
        self.importance_channels = importance_channels
        self.importance_activation = importance_activation
        self.importance_dropout_rate = importance_dropout_rate
        self.final_units = final_units
        self.final_activation = final_activation
        self.sparsity_factor = sparsity_factor
        self.exclusivity_factor = exclusivity_factor
        self.lay_additional_cb = lay_additional_cb

        self.lay_concat_input = LazyConcatenate(axis=-1)

        # This list contains all the actual GAT-like layers which do the graph convolutions. How many
        # layers there will be and how many hidden units they have is determined by the "self.units" list.
        self.attention_layers: List[GraphBaseLayer] = []
        for units in self.units:
            # This is a custom layer, which is essentially a modification of GATv2: It is actually K
            # separate attention channels, where the channels share the main conv weights, but each have
            # separate attention weights. Basically, there is one such parallel GAT channel for the number
            # of importance channels - in each layer.
            lay = MultiChannelGatLayer(
                units=units,
                channels=importance_channels,
                use_edge_features=use_edge_features,
                activation=activation,
                use_bias=use_bias,
                use_final_activation=True,
                has_self_loops=True,
            )
            self.attention_layers.append(lay)

        # various layers just for regularization
        self.lay_dropout = DropoutEmbedding(rate=dropout_rate)
        self.lay_dropout_importances = DropoutEmbedding(rate=importance_dropout_rate)
        self.lay_sparsity = ExplanationSparsityRegularization(coef=sparsity_factor)
        self.lay_exclusivity = ExplanationExclusivityRegularization(coef=exclusivity_factor)

        # ~ Edge Importances
        # The edge importances are derived from the edge attention weights which the GAT layers maintain
        # anyways. Since we have as many parallel GAT "heads" as there are to be importance channels, we
        # only have to reduce over the layer-dimension to get the edge importance values.
        self.lay_concat_alphas = LazyConcatenate(axis=-1)

        self.lay_pool_edges_in = PoolingLocalEdges(pooling_method='sum', pooling_index=0)
        self.lay_pool_edges_out = PoolingLocalEdges(pooling_method='sum', pooling_index=1)
        self.lay_average = LazyAverage()

        # ~ Node Importances
        # The node importances are calculated from an additional dense layer which works on the final
        # node embeddings that were produced by all of the previously defined GAT layers together.
        # The activation has to be linear since we apply the sigmoid afterwards manually. Also we need
        # one "embedding" for each importance channel in the end.
        self.lay_node_importances = DenseEmbedding(
            units=importance_channels,
            activation='linear',
            use_bias=False,
            kernel_initializer=ks.initializers.Ones(),
        )

        # ~ Output
        # The main GAT layers produce final node embeddings for the graph at the end (actually K separate
        # embedding vectors for K importance channels). These node embeddings are then globally pooled into
        # a graph embedding vector and on top of each of those a dense network then produces the final
        # predictions. Here is is important to note, that each channel does this separately: Each importance
        # channel only works on it's own node embeddings and also has it's own dense final network
        self.lay_pool_out = PoolingWeightedNodes

        self.final_units = final_units + [1]
        self.final_activations = [final_activation for _ in final_units] + ['linear']
        self.final_layers = []
        for i in range(importance_channels):
            # This final dense network may also have a certain depth, which is determined by the
            # list argument "final_units"
            layers = []
            for units, activation in zip(self.final_units, self.final_activations):
                lay = DenseEmbedding(
                    units=units,
                    activation=activation,
                    use_bias=use_bias,
                )
                layers.append(lay)

            self.final_layers.append(layers)

        self.lay_concat_out = LazyConcatenate(axis=-1)
        self.lay_final_activation = ActivationEmbedding(final_activation)

        # This is an optional final layer which the user may provide through a callback function which acts
        # as a factory for the desired layer. This is important for example if we want to do regression with
        # this kind of model. Because by default the innermost dimension of the output vector will be the
        # number of importance channels, whereas for regression it would have to be a single value. In that
        # case we then need a final layer which does this final reduction.
        self.lay_additional = None
        if self.lay_additional_cb is not None:
            self.lay_additional = self.lay_additional_cb()

    def call(self, inputs):
        # node_input: ([batch], [N], V)
        # edge_input: ([batch], [M], F)
        # edge_index_input: ([batch], [M], 2)
        node_input, edge_input, edge_index_input = inputs

        # Assuming K importance channels
        # turning input into shape ([batch], [N], K, V) as that is the expected input format for the
        # MultiChannelGatLayer
        node_input_expanded = self.lay_concat_input([tf.expand_dims(node_input, axis=-2)
                                                     for _ in range(self.importance_channels)])

        # Then we pass the input through all those multi channel gat layers. Basically each channel maintains
        # it's own node feature embeddings. Those are only combined into a single node embedding vector at
        # the very end. Throughout all the layers we also collect the edge attention coefficient "alpha"
        # vector
        alphas = []
        x = node_input_expanded
        for lay in self.attention_layers:
            # x: ([batch], [N], K, V)
            # alpha: ([batch], [M], K, 1)
            x, alpha = lay([x, edge_input, edge_index_input])
            alphas.append(alpha)

        # ~ edge importances
        # alphas: ([batch], [M], K, L) - where L is number of layers. This is also the dimension we reduce
        alphas = self.lay_concat_alphas(alphas)
        # edge_importances: ([batch], [M], K)
        edge_importances = tf.reduce_sum(alphas, axis=-1, keepdims=False)
        # We want importances to also be a [0, 1] attention-like value
        edge_importances = ks.activations.sigmoid(edge_importances)
        # Applying all the various regularization layers
        edge_importances = self.lay_dropout_importances(edge_importances)
        self.lay_exclusivity(edge_importances)
        self.lay_sparsity(edge_importances)

        # Now we need to pool the edge importances, which essentially turns them into node embeddings.

        # The first argument "node_input" we pass here is completely arbitrary. It does not even have
        # an effect.
        pooled_edges_in = self.lay_pool_edges_in([node_input, edge_importances, edge_index_input])
        pooled_edges_out = self.lay_pool_edges_out([node_input, edge_importances, edge_index_input])
        # pooled_edges: ([batch], [N], K)
        pooled_edges = self.lay_average([pooled_edges_in, pooled_edges_out])

        # ~ node_importances
        x = self.lay_dropout(x)
        x_combined = tf.reduce_sum(x, axis=-2)
        # node_importances: ([batch], [N], K)
        node_importances = self.lay_node_importances(x_combined)
        node_importances = ks.activations.sigmoid(node_importances)

        # This multiplication of node_importances and pooled edge_importances helps to make the explanation
        # more "consistent" -> There will likely be not too many freely floating node / edge importances.
        # Like this it is more likely that node and edges which are directly besides each other have similar
        # importance values.
        node_importances = node_importances * pooled_edges
        # Applying all the regularization
        self.lay_sparsity(node_importances)
        self.lay_exclusivity(node_importances)

        # ~ output
        outs = []
        for k in range(self.importance_channels):
            # For each channel we use the corresponding importance vector and use it as the weights to
            # perform a weighted global sum pooling on the corresponding node embeddings to produce the
            # overall graph embeddings
            node_importance_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)
            # ! For the node embedding vectors, the second last dimension is the importance channel dim!
            x_slice = x[:, :, k, :]
            # This pooling operation produces global graph embeddings
            # out: ([batch], V)
            out = self.lay_pool_out([x_slice, node_importance_slice])

            # Each channel also has it's own final dense output network, through which we propagate these
            # graph embeddings
            layers = self.final_layers[k]
            for lay in layers:
                out = lay(out)

            outs.insert(0, out)

        # out: ([batch], K)
        out = self.lay_concat_out(outs)
        out = self.lay_final_activation(out)

        if self.lay_additional is not None:
            out = self.lay_additional(out)

        return out, node_importances, edge_importances
