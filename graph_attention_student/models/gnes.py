import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingNodes

from graph_attention_student.layers import ExplanationSparsityRegularization
from graph_attention_student.models.gradient import AbstractGradientModel


class GnesGradientModel(AbstractGradientModel):

    def __init__(self,
                 units: t.List[int],
                 batch_size: int,
                 importance_func: t.Callable,
                 layer_cb: t.Callable = lambda units: GCN(units=units, activation='kgcnn>leaky_relu'),
                 sparsity_factor: float = 0.1,
                 final_units: t.List[int] = [1],
                 final_activation: str = 'linear',
                 pooling_method: str = 'sum'):
        super(GnesGradientModel, self).__init__()
        self.importance_func = importance_func
        self.batch_size = batch_size
        self.num_outputs = final_units[-1]

        self.conv_layers = []
        for k in units:
            lay = layer_cb(k)
            self.conv_layers.append(lay)

        # ~ global pooling
        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)

        # ~ mlp tail-end
        self.final_units = final_units
        self.final_acts = ['kgcnn>leaky_relu' for _ in self.final_units]
        self.final_acts[-1] = final_activation
        self.final_layers = []
        for k, act in zip(self.final_units, self.final_acts):
            lay = DenseEmbedding(
                units=k,
                activation=act
            )
            self.final_layers.append(lay)

        # ~ regularization
        self.lay_sparsity = ExplanationSparsityRegularization(factor=sparsity_factor)

    def call_with_gradients(self, inputs, training=True, batch_size=None, create_gradients=True):
        if batch_size is None:
            batch_size = self.batch_size

        node_input, edge_input, edge_index_input = inputs
        x = node_input

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(node_input)
            tape.watch(edge_input)

            # ~ convolutional layers
            edge_activations = [edge_input]
            node_activations = [node_input]
            for lay in self.conv_layers:
                x = lay([x, edge_input, edge_index_input])
                node_activations.append(x)

            # ~ global pooling
            out = self.lay_pooling(x)

            # ~ mlp tail end
            for lay in self.final_layers:
                out = lay(out)

            outs = [[out[b, o] for o in range(self.num_outputs)] for b in range(batch_size)]

        if create_gradients:
            node_gradient_info = self.calculate_gradient_info(outs, node_activations, tape, batch_size)
            edge_gradient_info = self.calculate_gradient_info(outs, edge_activations, tape, batch_size)

            node_importances = self.importance_func(node_gradient_info)
            edge_importances = self.importance_func(edge_gradient_info)

            self.lay_sparsity(node_importances)
            self.lay_sparsity(edge_importances)

            return out, node_importances, edge_importances

        else:
            return out, None, None

    def call(self, inputs, *args, **kwargs):
        return self.call_with_gradients(inputs, *args, **kwargs)