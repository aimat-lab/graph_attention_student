from typing import List, Optional, Callable, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.python.keras.engine import compile_utils
from kgcnn.layers.modules import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate, LazyAverage
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.pooling import PoolingWeightedNodes
from kgcnn.utils.data import ragged_tensor_from_nested_numpy

from graph_attention_student.layers import MultiChannelGatLayer
from graph_attention_student.layers import MultiHeadGatLayer
from graph_attention_student.layers import ExplanationSparsityRegularization
from graph_attention_student.layers import ExplanationExclusivityRegularization
from graph_attention_student.layers import StaticMultiplicationEmbedding
from graph_attention_student.training import RecompilableMixin
from graph_attention_student.training import shifted_sigmoid


class MultiAttentionStudent(RecompilableMixin, ks.models.Model):

    def __init__(self,
                 # primary network related arguments
                 units: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 use_edge_features: bool = True,
                 # importance related arguments
                 importance_units: List[int] = [],
                 importance_channels: int = 2,
                 importance_activation: str = 'sigmoid',
                 importance_dropout_rate: float = 0.0,
                 final_units: List[int] = [],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'softmax',
                 importance_factor: float = 0.0,
                 regression_limits: Optional[Union[Tuple[float, float], List[tuple]]] = None,
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        RecompilableMixin.__init__(self)

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_edge_features = use_edge_features

        self.regression_limits: Optional[Tuple[float, float]] = regression_limits
        self.regression_center: Optional[float] = None

        self.importance_channels = importance_channels
        self.importance_activation = importance_activation
        self.importance_dropout_rate = importance_dropout_rate
        self.final_units = final_units
        self.final_activation = final_activation
        self.importance_factor = importance_factor

        # This list contains all the actual GAT-like layers which do the graph convolutions. How many
        # layers there will be and how many hidden units they have is determined by the "self.units" list.
        self.attention_layers: List[GraphBaseLayer] = []
        for units in self.units:
            # This is a custom layer, which is essentially a multi-head GATv2, just compiled into a single
            # layer.
            lay = MultiHeadGatLayer(
                units=units,
                num_heads=importance_channels,
                use_edge_features=use_edge_features,
                activation=activation,
                use_bias=use_bias,
                use_final_activation=True,
                has_self_loops=True,
                share_weights=False,
                # If this is True the embeddings of the heads are concat-ed and then passed to the input of
                # the next layer. If it is False, they are averaged instead
                concat_heads=True,
            )
            self.attention_layers.append(lay)

        # various layers just for regularization
        self.lay_dropout = DropoutEmbedding(rate=dropout_rate)
        self.lay_dropout_importances = DropoutEmbedding(rate=importance_dropout_rate)

        self.lay_act_importance = ActivationEmbedding(importance_activation)

        # ~ Edge Importances
        # The edge importances are derived from the edge attention weights which the GAT layers maintain
        # anyways. Since we have as many parallel GAT "heads" as there are to be importance channels, we
        # only have to reduce over the layer-dimension to get the edge importance values.
        self.lay_concat_alphas = LazyConcatenate(axis=-1)

        self.lay_pool_edges_in = PoolingLocalEdges(pooling_method='mean', pooling_index=0)
        self.lay_pool_edges_out = PoolingLocalEdges(pooling_method='mean', pooling_index=1)
        self.lay_average = LazyAverage()

        # ~ Node Importances
        # The node importances are calculated from an additional series of dense layer which works on
        # the final node embedding that was produced by all of the previously defined GAT layers together.

        # The final unit count has to be exactly the number of importance channels.
        self.node_importance_units = importance_units + [importance_channels]
        # The final activation is a relu here, which might seem weird because we additionally also use an
        # additional sigmoid activation on this already relu-ed value. This is actually really important
        # and further explained in the docstring.
        self.node_importance_acts = ['relu' for _ in importance_units] + ['linear']
        self.node_importance_layers = []
        for k, act in zip(self.node_importance_units, self.node_importance_acts):
            lay = DenseEmbedding(
                units=k,
                activation=act,
                use_bias=use_bias,
            )
            self.node_importance_layers.append(lay)

        # ~ Output
        # The main GAT layers produce final node embeddings for the graph at the end
        # These node embeddings are then globally pooled into a vector of graph embeddings - actually one
        # such vector of graph embeddings for each importance channel. Those are then concat-ed and fed into
        # a final series of Dense layers, which then produce the final output...
        self.lay_pool_out_weighted = PoolingWeightedNodes(pooling_method='sum')
        self.lay_pool_out = PoolingNodes(pooling_method='sum')

        self.lay_concat_out = LazyConcatenate(axis=-1)
        self.lay_final_dropout = DropoutEmbedding(final_dropout_rate)
        self.lay_final_activation = ActivationEmbedding(final_activation)

        self.out_units = final_units + [1]
        self.out_activations = ['relu' for _ in final_units] + ['linear']
        self.out_layers = []
        for k, act in zip(self.out_units, self.out_activations):
            lay = DenseEmbedding(
                units=k,
                activation=act,
                use_bias=use_bias,
            )
            self.out_layers.append(lay)

        # tbd
        self.bce_loss = ks.losses.BinaryCrossentropy()
        self.compiled_bce_loss = compile_utils.LossesContainer(self.bce_loss)

        self.mse_loss = ks.losses.MeanSquaredError()
        self.compiled_mse_loss = compile_utils.LossesContainer(self.mse_loss)

        self.lay_exp = DenseEmbedding(
            units=1,
            use_bias=False,
            kernel_constraint=ks.constraints.NonNeg(),
        )

        if self.doing_classification:
            self.lay_final = DenseEmbedding(units=self.importance_channels, activation='softmax')
        elif self.doing_regression:
            self.lay_final = DenseEmbedding(units=1, activation='linear')

        # By default the network is built to do multi-class graph classification, but it is also possible
        # to do regression, but that requires some special changes.
        if self.regression_limits is not None:
            self.regression_center = np.mean(self.regression_limits)
            self.regression_width = np.abs(self.regression_limits[1] - self.regression_limits[0])

    @property
    def doing_regression(self) -> bool:
        return self.regression_limits is not None

    @property
    def doing_classification(self) -> bool:
        return self.regression_limits is None

    def finalize(self, out):
        pass

    def call(self, inputs, training=True, **kwargs):
        # node_input: ([batch], [N], V)
        # edge_input: ([batch], [M], F)
        # edge_index_input: ([batch], [M], 2)
        node_input, edge_input, edge_index_input = inputs

        # Then we pass the input through all those multi channel gat layers. Basically each channel maintains
        # it's own node feature embeddings. Those are only combined into a single node embedding vector at
        # the very end. Throughout all the layers we also collect the edge attention coefficient "alpha"
        # vector
        alphas = []
        xs = []
        x = node_input
        for lay in self.attention_layers:
            # x: ([batch], [N], V)
            # alpha: ([batch], [M], K, 1)
            x = self.lay_dropout(x)
            x, alpha = lay([x, edge_input, edge_index_input])

            xs.append(x)
            alphas.append(alpha)

        # ~ edge importances
        # alphas: ([batch], [M], K, L) - where L is number of layers. This is also the dimension we reduce
        alphas = self.lay_concat_alphas(alphas)
        alphas = tf.reduce_sum(alphas, axis=-1, keepdims=False)
        # edge_importances: ([batch], [M], K)
        edge_importances = alphas
        # We want importances to also be a [0, 1] attention-like value
        edge_importances = ks.activations.sigmoid(edge_importances)

        # Now we need to pool the edge importances, which essentially turns them into node embeddings.

        # The first argument "node_input" we pass here is completely arbitrary. It does not even have
        # an effect.
        pooled_edges_in = self.lay_pool_edges_in([node_input, edge_importances, edge_index_input])
        pooled_edges_out = self.lay_pool_edges_out([node_input, edge_importances, edge_index_input])
        # pooled_edges: ([batch], [N], K)
        pooled_edges = self.lay_average([pooled_edges_in, pooled_edges_out])

        # ~ node_importances
        node_importances_raw = tf.concat([x], axis=-1)
        for lay in self.node_importance_layers:
            node_importances_raw = lay(node_importances_raw)

        node_importances = ks.activations.sigmoid(node_importances_raw)
        #node_importances = shifted_sigmoid(node_importances_raw)

        # This multiplication of node_importances and pooled edge_importances helps to make the explanation
        # more "consistent" -> There will likely be not too many freely floating node / edge importances.
        # Like this it is more likely that node and edges which are directly besides each other have similar
        # importance values.
        node_importances = node_importances * pooled_edges

        # ~ output
        outs = []
        for k in range(self.importance_channels):
            # For each channel we use the corresponding importance vector and use it as the weights to
            # perform a weighted global sum pooling on the corresponding node embeddings to produce the
            # overall graph embeddings
            node_importance_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)

            # This pooling operation produces global graph embeddings
            # out: ([batch], V, 1)
            out = self.lay_pool_out(x * node_importance_slice)
            # for lay in self.out_layers[k]:
            #     out = lay(out)

            outs.append(out)

        # out: ([batch], K)
        out = self.lay_concat_out(outs)

        for lay in self.out_layers:
            out = self.lay_final_dropout(out)
            out = lay(out)

        if self.doing_regression:
            out = tf.squeeze(out, axis=-1)

        if training:
            return out, x, node_importances, edge_importances
        else:
            return out, node_importances, edge_importances

    def predict_single(self, inputs: Tuple[List[float], List[float], List[float]]):
        node_input, edge_input, edge_index_input = inputs

        # The input is just normal lists, to be processable by the model we need to turn those into
        # ragged tensors first
        label, node_importances, edge_importances = self([
            ragged_tensor_from_nested_numpy(np.array([node_input])),
            ragged_tensor_from_nested_numpy(np.array([edge_input])),
            ragged_tensor_from_nested_numpy(np.array([edge_index_input]))
        ], training=False, finalize=True)

        # The results come out as tensors as well, we convert them back to lists.
        return (
            label.numpy().tolist()[0],
            node_importances.to_list()[0],
            edge_importances.to_list()[0]
        )

    def regression_augmentation(self,
                                out_true: tf.RaggedTensor):
        center_distances = tf.abs(out_true - self.regression_center)

        lo_samples = tf.where(out_true < self.regression_center, center_distances,  0.0)
        lo_samples = tf.expand_dims(lo_samples, axis=-1)

        hi_samples = tf.where(out_true >= self.regression_center, center_distances,  0.0)
        hi_samples = tf.expand_dims(hi_samples, axis=-1)

        reg_true = tf.concat([lo_samples, hi_samples], axis=-1)
        return reg_true

    def train_step_explanation(self,
                               x: Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor],
                               y: Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor],
                               ) -> None:
        """
        Given the input of one batch ``x`` and the corresponding ground truth values ``y`` performs a single
        complete train step for the explanation. Calculates a gradient and applies changes immediately to
        the trainable parameters.

        Generally, the explanation is trained by implicitly assuming that the GAT part of the
        network outputs a constant value of 1 for every node embedding. With this assumption the same global
        pooling is performed to create graph embeddings for every explanation channel through a weighted
        global pooling with the corresponding node importance channel values. The additional output dense
        layers are then discarded for this step.

        These graph embeddings are then used to solve an augmented problem, which depends on whether the
        network is doing regression or classification.

        **CLASSIFICATION:**

        The augmented problem uses the same ground truth class labels. The previously described graph
        embeddings for each channel are submitted to a modified sigmoid activation and then a BCE loss is
        used to train each channel to individually predict one of the classes.

        **REGRESSION:**

        For regression, we first of all need a-priori knowledge about the range of possible values which are
        represented in the dataset. This is used to deduce the *center* of this value range.

        :param x: Tuple consisting of the 3 required input tensors: <br>
            - node_input: ragged tensor of node features ([batch], [V], N0) <br>
            - edge_input: ragged tensor of edge features ([batch], [E], M0) <br>
            - edge_indices: ragged tensor of edge index tuples ([batch], [E], 2)
        :param y: Tuple consisting of ground truth target tensors: <br>
            - out_true: ragged tensor of shape ([batch], ?). Final dimension depends on whether the network
              is used for regression or classification. <br>
            - ni_true: ragged tensor of node importances ([batch], [V], K) <br>
            - ei_true: ragged tensor of edge importances ([batch], [E], K)
        :return: None
        """
        out_true, ni_true, ei_true = y

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            out_pred, x_pred, ni_pred, ei_pred = y_pred

            outs = []
            for k in range(self.importance_channels):
                node_importance_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)
                out = self.lay_pool_out(node_importance_slice)
                outs.append(out)

            out = self.lay_concat_out(outs)

            if self.doing_regression:

                reg_true = self.regression_augmentation(out_true)
                reg_pred = ks.backend.relu(out)
                exp_loss = self.compiled_mse_loss(reg_true, reg_pred)

            elif self.doing_classification:

                class_pred = shifted_sigmoid(out)
                exp_loss = self.compiled_bce_loss(out_true, class_pred)

            exp_loss *= self.importance_factor

        # Compute gradients
        trainable_vars = self.trainable_variables
        exp_gradients = tape.gradient(exp_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(exp_gradients, trainable_vars))

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # ~ EXPLANATION TRAINING
        # "train_explanation" actually performs an entire training step just for the attention "explanation"
        # part of the network. Look up the details in the function. The gist is that it only uses part of
        # the network to produce a simplified output which is then used to create a single training step
        # for the explanation part of the network.
        self.train_explanation(x, y)

        # ~ PREDICTION TRAINING
        # This performs a "normal" training step, as it is in the default implementation of the "train_step"
        out_true, ni_true, ei_true = y
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            out_pred, x_pred, ni_pred, ei_pred = y_pred

            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                [out_true, ni_true, ei_true],
                [out_pred, ni_pred, ei_pred],
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(
            [out_true, ni_true, ei_true],
            [out_pred, ni_pred, ei_pred],
            sample_weight=sample_weight
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
