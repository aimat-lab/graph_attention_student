"""
DEPRECATED - The contents of this module have been split into individual modules within the "models"
package.
"""
from typing import List, Optional, Callable, Tuple, Union
import time
import logging
import typing as t

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.python.keras.engine import compile_utils
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.modules import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate, LazyAverage
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.pooling import PoolingWeightedNodes
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.conv.gat_conv import AttentionHeadGAT, AttentionHeadGATV2
from kgcnn.literature.GNNExplain import GNNInterface

from graph_attention_student.layers import MultiHeadGatLayer
from graph_attention_student.layers import MultiHeadGATV2Layer
from graph_attention_student.layers import ExplanationSparsityRegularization
from graph_attention_student.training import RecompilableMixin
from graph_attention_student.training import shifted_sigmoid
from graph_attention_student.training import mae, mse
from graph_attention_student.util import NULL_LOGGER


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
                 importance_multiplier: float = 1,
                 importance_sparsity: float = 1e-3,
                 importance_exclusivity: float = 0.0,
                 final_units: List[int] = [],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'softmax',
                 importance_factor: float = 0.0,
                 regression_limits: Optional[Union[Tuple[float, float], List[tuple]]] = None,
                 regression_bins: Optional[List[List[float]]] = None,
                 regression_reference: float = 0,
                 sparsity_factor: float = 0.0,
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        RecompilableMixin.__init__(self)

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_edge_features = use_edge_features

        self.regression_reference = regression_reference
        self.regression_limits: Optional[Tuple[float, float]] = regression_limits
        self.regression_bins = regression_bins
        self.regression_center: Optional[float] = None

        self.importance_multiplier = importance_multiplier
        self.importance_channels = importance_channels
        self.importance_activation = importance_activation
        self.importance_dropout_rate = importance_dropout_rate
        self.final_units = final_units
        self.final_activation = final_activation
        self.importance_factor = importance_factor
        self.importance_exclusivity = importance_exclusivity
        if isinstance(importance_sparsity, float):
            self.importance_sparsity = [importance_sparsity for _ in range(importance_channels)]
        elif isinstance(importance_sparsity, list):
            self.importance_sparsity = importance_sparsity

        # ~ VALIDATING ARGUMENTS ~

        # ~ NETWORK COMPONENTS ~

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

        if self.doing_regression:
            self.out_units = final_units + [1]
            self.out_activations = ['relu' for _ in final_units] + ['linear']
            self.lay_final_activation = ActivationEmbedding('linear')

        elif self.doing_classification:
            self.out_units = final_units + [self.importance_channels]
            self.out_activations = ['relu' for _ in final_units] + ['linear']
            self.lay_final_activation = ActivationEmbedding('softmax')

        self.out_layers = []
        for k, act in zip(self.out_units, self.out_activations):
            lay = DenseEmbedding(
                units=k,
                activation=act,
                use_bias=use_bias,
                # use_bias=False
            )
            self.out_layers.append(lay)

        # tbd
        self.bce_loss = ks.losses.BinaryCrossentropy()
        self.compiled_bce_loss = compile_utils.LossesContainer(self.bce_loss)

        self.cce_loss = ks.losses.CategoricalCrossentropy()
        self.compiled_cce_loss = compile_utils.LossesContainer(self.cce_loss)

        self.mse_loss = ks.losses.MeanSquaredError()
        self.compiled_mse_loss = compile_utils.LossesContainer(self.mse_loss)

        self.lay_exp = DenseEmbedding(
            units=self.importance_channels,
            use_bias=False,
            #kernel_constraint=ks.constraints.NonNeg(),
            kernel_initializer=ks.initializers.Identity(),
        )
        self.exp_layers = []
        for k in range(self.importance_channels):
            lay = DenseEmbedding(
                units=1,
                activation='relu',
                use_bias=True
            )
            self.exp_layers.append(lay)

        if self.doing_classification:
            self.lay_final = DenseEmbedding(units=self.importance_channels, activation='softmax')
        elif self.doing_regression:
            self.lay_final = DenseEmbedding(units=1, activation='linear')

        # By default the network is built to do multi-class graph classification, but it is also possible
        # to do regression, but that requires some special changes.
        if self.regression_limits is not None:
            self.regression_center = self.regression_reference
            self.regression_width = np.abs(self.regression_limits[1] - self.regression_limits[0])

        self.lay_sparsity = ExplanationSparsityRegularization(coef=sparsity_factor)

    # def build(self, *args, **kwargs):
    #     print('BUILD', args, kwargs)
    #     super(MultiAttentionStudent, self).build(*args, **kwargs)

    @property
    def doing_regression(self) -> bool:
        return self.regression_limits is not None

    @property
    def doing_classification(self) -> bool:
        return self.regression_bins is None

    def finalize(self, out):
        pass

    def call(self,
             inputs,
             training=False,
             external_node_importances: Optional[np.ndarray] = None,
             mask_channel: Optional[int] = None,
             **kwargs):
        # node_input: ([batch], [N], V)
        # edge_input: ([batch], [M], F)
        # edge_index_input: ([batch], [M], 2)
        node_input, edge_input, edge_index_input = inputs
        #print(node_input.shape, edge_input.shape, edge_index_input.shape)

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
            x, alpha = lay([x, edge_input, edge_index_input])
            if training:
                x = self.lay_dropout(x, training=training)

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
        node_importances_slices = []

        # This multiplication of node_importances and pooled edge_importances helps to make the explanation
        # more "consistent" -> There will likely be not too many freely floating node / edge importances.
        # Like this it is more likely that node and edges which are directly besides each other have similar
        # importance values.
        node_importances = node_importances * pooled_edges

        for k in range(self.importance_channels):
            importances_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)
            node_importances_slices.append(importances_slice)

        if mask_channel is not None:
            node_importances_slices = [tf.zeros_like(s) if k != mask_channel else s
                                       for k, s in enumerate(node_importances_slices)]

        node_importances = tf.concat(node_importances_slices, axis=-1)
        self.lay_sparsity(node_importances)
        node_importances = self.lay_dropout_importances(node_importances)

        if external_node_importances is not None:
            node_importances = external_node_importances

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
            out = lay(out)
            if training:
                out = self.lay_final_dropout(out, training=training)

        logits = out
        out = self.lay_final_activation(logits)

        if self.doing_regression:
            out = tf.squeeze(out, axis=-1)
            out = out + self.regression_reference

        if training:
            return out, logits, x, node_importances, edge_importances
        else:
            return out, node_importances, edge_importances

    def predict_single(self,
                       inputs: Tuple[List[float], List[float], List[float]],
                       external_node_importances: Optional[List[List[float]]] = None):
        node_input, edge_input, edge_index_input = inputs

        if external_node_importances is not None:
            external_node_importances = ragged_tensor_from_nested_numpy(np.array([external_node_importances]))
            external_node_importances = tf.cast(external_node_importances, tf.float32)

        # The input is just normal lists, to be processable by the model we need to turn those into
        # ragged tensors first
        label, node_importances, edge_importances = self([
                ragged_tensor_from_nested_numpy(np.array([node_input])),
                ragged_tensor_from_nested_numpy(np.array([edge_input])),
                ragged_tensor_from_nested_numpy(np.array([edge_index_input]))
            ],
            training=False,
            finalize=True,
            external_node_importances=external_node_importances
        )

        # The results come out as tensors as well, we convert them back to lists.
        return (
            label.numpy().tolist()[0],
            node_importances.to_list()[0],
            edge_importances.to_list()[0]
        )

    def regression_augmentation(self,
                                out_true: tf.RaggedTensor):
        center_distances = tf.abs(out_true - self.regression_center)
        center_distances = (center_distances * self.importance_multiplier) / (0.5 * self.regression_width)

        lo_samples = tf.where(out_true < self.regression_center, center_distances,  0.0)
        lo_samples = tf.expand_dims(lo_samples, axis=-1)
        lo_mask = tf.where(out_true < self.regression_center, 1.0, 0.0)
        lo_mask = tf.expand_dims(lo_mask, axis=-1)

        hi_samples = tf.where(out_true > self.regression_center, center_distances,  0.0)
        hi_samples = tf.expand_dims(hi_samples, axis=-1)
        hi_mask = tf.where(out_true > self.regression_center, 1.0, 0.0)
        hi_mask = tf.expand_dims(hi_mask, axis=-1)

        reg_true = tf.concat([lo_samples, hi_samples], axis=-1)
        mask = tf.concat([lo_mask, hi_mask], axis=-1)
        # reg_true = tf.concat([hi_samples, lo_samples], axis=-1)
        # mask = tf.concat([hi_mask, lo_mask], axis=-1)

        return reg_true, mask

    def create_regression_mask(self,
                               out_true: tf.RaggedTensor):

        mask = []
        for k, (lower, upper) in enumerate(self.regression_bins):
            mask_slice = tf.where(out_true <= upper, 1.0, 0.0) * tf.where(out_true >= lower, 1.0, 0.0)
            mask_slice = tf.expand_dims(mask_slice, axis=-1)
            mask.append(mask_slice)

        mask = tf.concat(mask, axis=-1)

        return mask

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

        For regression we first need a-priori knowledge about the range of values to be expected from the
        dataset. From that we can deduce the center of that value range. By default we map a regression
        problem to be composed of 2 channels: One which explains high values and one which explains low
        values (relative to the center of the expected value range).
        We then set up two separate regression problems by only considering the absolute distance from this
        center value. One of the channels is then only trained on all the samples which have values below
        the center value and the other is only trained with the samples which have original values above
        this center value.

        .. code-block:: text

            # pseudocode
            lo_true = abs(y_true - center) where y_true < center
            hi_true = abs(y_true - center) where y_true >= center

            reg_true = concat(lo_true, hi_true)

        The predictions of the network are computed by applying a relu activation on the previously described
        simplified graph embeddings of each channel and then using MSE loss to train on the vector of
        constructed regression values ``reg_true``.

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
        node_input, edge_input, edge_index_input = x
        out_true, ni_true, ei_true = y

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            out_pred, logits_pred, x_pred, ni_pred, ei_pred = y_pred

            outs = []
            for k in range(self.importance_channels):
                node_importance_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)

                out = self.lay_pool_out(node_importance_slice)
                # out = self.lay_pool_out(node_importance_slice * x_pred)
                # out = tf.reduce_sum(out, keepdims=1)

                #out = self.exp_layers[k](out)

                outs.append(out)

            out = self.lay_concat_out(outs)

            if self.doing_regression:

                reg_true, mask = self.regression_augmentation(out_true)
                reg_pred = out

                exp_loss = (self.importance_channels) * \
                           self.compiled_mse_loss(reg_true * mask, reg_pred * mask)

                exp_loss += (self.importance_sparsity[k] / self.importance_channels) \
                    * tf.reduce_mean(tf.abs(ni_pred[:, :, k]))

            elif self.doing_classification:
                class_pred = shifted_sigmoid(out, multiplier=self.importance_multiplier)
                exp_loss = self.compiled_bce_loss(out_true, class_pred)

            exp_loss *= self.importance_factor

        # Compute gradients
        trainable_vars = self.trainable_variables
        exp_gradients = tape.gradient(exp_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(exp_gradients, trainable_vars))

        return {'exp_loss': exp_loss}

    def train_step_explanation_alt(self,
                                   x: Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor],
                                   y: Tuple[tf.RaggedTensor, tf.RaggedTensor, tf.RaggedTensor],
                                   ) -> None:
        node_input, edge_input, edge_index_input = x
        out_true, ni_true, ei_true = y

        if self.doing_regression:
            mask_reg = self.create_regression_mask(out_true)

        exp_loss = 0.0
        with tf.GradientTape(persistent=True) as tape:
            ni_pred_combined = []
            for k in range(self.importance_channels):

                y_pred = self(x, training=True, mask_channel=k)  # Forward pass
                out_pred, logits_pred, x_pred, ni_pred, ei_pred = y_pred

                if self.doing_regression:
                    # reg_pred = self.lay_exp(out)
                    # exp_loss += self.compiled_mse_loss(
                    #     tf.expand_dims(reg_true[:, k], axis=-1),
                    #     tf.expand_dims(out_pred, axis=-1),
                    #     sample_weight=mask[:, k]
                    # )
                    exp_loss += self.compiled_mse_loss(
                        tf.expand_dims(out_true, axis=-1),
                        tf.expand_dims(out_pred, axis=-1),
                        sample_weight=mask_reg[:, k]
                    )

                    exp_loss += (self.importance_sparsity[k] / self.importance_channels) \
                        * tf.reduce_mean(tf.abs(ni_pred[:, :, k]))

                    ni_pred_combined.append(tf.expand_dims(ni_pred[:, :, k], axis=-1))

                elif self.doing_classification:
                    out_true = tf.cast(out_true, tf.float32)
                    mask = tf.where(out_true[:, k] > 0.9, 1.0, 0.0)
                    # mask = tf.expand_dims(mask, axis=-1)
                    # exp_loss += self.compiled_cce_loss(out_true, out_pred, sample_weight=mask)

                    class_pred = tf.expand_dims(ks.activations.sigmoid(logits_pred[:, k]), axis=-1)
                    class_true = tf.ones_like(class_pred)

                    exp_loss += self.compiled_bce_loss(class_true, class_pred, sample_weight=mask)
                    exp_loss += (self.importance_sparsity[k] / self.importance_channels) \
                        * tf.reduce_mean(tf.abs(ni_pred[:, :, k]))

                    ni_pred_combined.append(tf.expand_dims(ni_pred[:, :, k], axis=-1))

            ni_pred_combined = tf.concat(ni_pred_combined, axis=-1)
            entropy = -tf.reduce_sum(ni_pred_combined * tf.math.log(tf.abs(ni_pred_combined) + 1e-8), axis=-1)
            exp_loss += self.importance_exclusivity * tf.reduce_mean(entropy)
            exp_loss *= self.importance_factor

        # Compute gradients
        trainable_vars = self.trainable_variables
        exp_gradients = tape.gradient(exp_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(exp_gradients, trainable_vars))

        return {'exp_loss': exp_loss}

    def train_step(self,
                   data: Tuple[tuple, tuple]
                   ) -> None:
        """
        Performs one training step with the given ``data`` consisting of the input data ``x`` of one batch
        as well as the corresponding ground truth labels ``y``.

        :param data: Tuple consisting of the input data x of a batch as well as the corresponding ground
            truth labels y. <br>
            x: Tuple consisting of the 3 required input tensors: <br>
            - node_input: ragged tensor of node features ([batch], [V], N0) <br>
            - edge_input: ragged tensor of edge features ([batch], [E], M0) <br>
            - edge_indices: ragged tensor of edge index tuples ([batch], [E], 2)
            y: Tuple consisting of ground truth target tensors: <br>
            - out_true: ragged tensor of shape ([batch], ?). Final dimension depends on whether the network
              is used for regression or classification. <br>
            - ni_true: ragged tensor of node importances ([batch], [V], K) <br>
            - ei_true: ragged tensor of edge importances ([batch], [E], K)
        :return: None
        """
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
        exp_metrics = {}
        if self.importance_channels != 1 and self.importance_factor != 0:
            exp_metrics = self.train_step_explanation(x, y)
            # exp_metrics = self.train_step_explanation_alt(x, y)
            pass

        # ~ PREDICTION TRAINING
        # This performs a "normal" training step, as it is in the default implementation of the "train_step"
        out_true, ni_true, ei_true = y
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            out_pred, logits_pred, x_pred, ni_pred, ei_pred = y_pred

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
        return {
            **{m.name: m.result() for m in self.metrics},
            **exp_metrics
        }


class GnnxGCN(ks.models.Model, GNNInterface):

    def __init__(self,
                 units: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 pooling_method: str = 'sum',
                 final_units: List[int] = [],
                 final_activation: str = 'linear',
                 outputs: int = 1,
                 **kwargs):
        ks.models.Model.__init__(self, **kwargs)
        GNNInterface.__init__(self)

        self.conv_layers = []
        for k in units:
            lay = GCN(
                units=k,
                activation=activation,
            )
            self.conv_layers.append(lay)

        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        self.final_activations = ['relu' for _ in final_units] + [final_activation]
        self.final_units = [k for k in final_units] + [outputs]
        self.final_layers = []
        for k, act in zip(self.final_units, self.final_activations):
            lay = DenseEmbedding(
                units=k,
                activation=act,
            )
            self.final_layers.append(lay)

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay in self.conv_layers:
            x = lay([x, edge_input, edge_index_input])

        out = self.lay_pooling(x)
        for lay in self.final_layers:
            out = lay(out)

        return out

    def predict(self, gnn_input):
        node_input, edge_input, edge_index_input = gnn_input

        return self([node_input, edge_input, edge_index_input], training=False)[0]

    def masked_predict(self, gnn_input, edge_mask, feature_mask, node_mask, **kwargs):
        node_input, edge_input, edge_index_input = gnn_input

        edge_mask = tf.RaggedTensor.from_tensor(tf.cast(tf.expand_dims(edge_mask, axis=0), dtype=tf.float32))
        node_mask = tf.RaggedTensor.from_tensor(tf.cast(tf.expand_dims(node_mask, axis=0), dtype=tf.float32))
        # print(edge_input.shape, edge_mask.shape)
        edge_input_masked = tf.cast(edge_input, dtype=tf.float32) * edge_mask
        node_input_masked = tf.cast(node_input, dtype=tf.float32) * node_mask

        return self([node_input_masked, edge_input_masked, edge_index_input], training=False)[0]

    def get_explanation(self, gnn_input, edge_mask, feature_mask, node_mask):
        return node_mask.numpy().tolist(), edge_mask.numpy().tolist()

    def get_number_of_nodes(self, gnn_input):
        node_input, _, _ = gnn_input
        return node_input[0].shape[0]

    def get_number_of_node_features(self, gnn_input):
        node_input, _, _ = gnn_input
        return node_input.shape[2]

    def get_number_of_edges(self, gnn_input):
        _, edge_input, _ = gnn_input
        return edge_input[0].shape[0]


class Megan(ks.models.Model):
    """
    MEGAN: Multi Explanation Graph Attention Network
    This model currently supports graph regression and graph classification problems. It was mainly designed
    with a focus on explainable AI (XAI). Along the main prediction, this model is able to output multiple
    attention-based explanations for that prediction. More specifically, the model outputs node and edge
    attributional explanations (assigning [0, 1] values to ever node / edge of the input graph) in K
    separate explanation channels, where K can be chosen as an independent model parameter.
    """

    def __init__(self,
                 # convolutional network related arguments
                 units: t.List[int],
                 activation: str = "kgcnn>leaky_relu",
                 use_bias: bool = True,
                 dropout_rate: float = 0.0,
                 use_edge_features: bool = True,
                 # node/edge importance related arguments
                 importance_units: t.List[int] = [],
                 importance_channels: int = 2,
                 importance_activation: str = "sigmoid",  # do not change
                 importance_dropout_rate: float = 0.0,  # do not change
                 importance_factor: float = 0.0,
                 importance_multiplier: float = 10.0,
                 importance_transformations: t.Optional[t.List[ks.layers.Layer]] = None,
                 sparsity_factor: float = 0.0,
                 concat_heads: bool = True,
                 separate_explanation_step: bool = False,
                 # mlp tail end related arguments
                 final_units: t.List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 final_pooling: str = 'sum',
                 regression_limits: t.Optional[t.Tuple[float, float]] = None,
                 regression_bins: t.Optional[t.List[t.Tuple[float, float]]] = None,
                 regression_reference: t.Optional[float] = None,
                 return_importances: bool = True,
                 use_graph_attributes: bool = False,
                 **kwargs):
        """
        Args:
            units: A list of ints where each element configures an additional attention layer. The numeric
                value determines the number of hidden units to be used in the attention heads of that layer
            activation: The activation function to be used within the attention layers of the network
            use_bias: Whether the layers of the network should use bias weights at all
            dropout_rate: The dropout rate to be applied after *each* of the attention layers of the network.
            use_edge_features: Whether edge features should be used. Generally the network supports the
                usage of edge features, but if the input data does not contain edge features, this should be
                set to False.
            importance_units: A list of ints where each element configures another dense layer in the
                subnetwork that produces the node importance tensor from the main node embeddings. The
                numeric value determines the number of hidden units in that layer.
            importance_channels: The int number of explanation channels to be produced by the network. This
                is the value referred to as "K". Note that this will also determine the number of attention
                heads used within the attention subnetwork.
            importance_factor: The weight of the explanation-only train step. If this is set to exactly
                zero then the explanation train step will not be executed at all (less computationally
                expensive)
            importance_multiplier: An additional hyperparameter of the explanation-only train step. This
                is essentially the scaling factor that is applied to the values of the dataset such that
                the target values can reasonably be approximated by a sum of [0, 1] importance values.
            sparsity_factor: The coefficient for the sparsity regularization of the node importance
                tensor.
            concat_heads: Whether to concat the heads of the attention subnetwork. The default is True. In
                that case the output of each individual attention head is concatenated and the concatenated
                vector is then used as the input of the next attention layer's heads. If this is False, the
                vectors are average pooled instead.
            final_units: A list of ints where each element configures another dense layer in the MLP
                at the tail end of the network. The numeric value determines the number of the hidden units
                in that layer. Note that the final element in this list has to be the same as the dimension
                to be expected for the samples of the training dataset!
            final_dropout_rate: The dropout rate to be applied after *every* layer of the final MLP.
            final_activation: The activation to be applied at the very last layer of the MLP to produce the
                actual output of the network.
            final_pooling: The pooling method to be used during the global pooling phase in the network.
            regression_limits: A tuple where the first value is the lower limit for the expected value range
                of the regression task and teh second value the upper limit.
            regression_reference: A reference value which is inside the range of expected values (best if
                it was in the middle, but does not have to). Choosing different references will result
                in different explanations.
            return_importances: Whether the importance / explanation tensors should be returned as an output
                of the model. If this is True, the output of the model will be a 3-tuple:
                (output, node importances, edge importances), otherwise it is just the output itself
        """
        ks.models.Model.__init__(self, **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_edge_features = use_edge_features
        self.importance_units = importance_units
        self.importance_channels = importance_channels
        self.importance_activation = importance_activation
        self.importance_dropout_rate = importance_dropout_rate
        self.importance_factor = importance_factor
        self.importance_multiplier = importance_multiplier
        self.importance_transformations = importance_transformations
        self.sparsity_factor = sparsity_factor
        self.concat_heads = concat_heads
        self.final_units = final_units
        self.final_dropout_rate = final_dropout_rate
        self.final_activation = final_activation
        self.final_pooling = final_pooling
        self.regression_limits = regression_limits
        self.regression_reference = regression_reference
        self.regression_bins = regression_bins
        self.return_importances = return_importances
        self.separate_explanation_step = separate_explanation_step
        self.use_graph_attributes = use_graph_attributes

        # ~ MAIN CONVOLUTIONAL / ATTENTION LAYERS
        self.attention_layers: t.List[GraphBaseLayer] = []
        for u in self.units:
            lay = MultiHeadGATV2Layer(
                units=u,
                num_heads=self.importance_channels,
                use_edge_features=self.use_edge_features,
                activation=self.activation,
                use_bias=self.use_bias,
                has_self_loops=True,
                concat_heads=self.concat_heads
            )
            self.attention_layers.append(lay)

        self.lay_dropout = DropoutEmbedding(rate=self.dropout_rate)
        self.lay_sparsity = ExplanationSparsityRegularization(factor=self.sparsity_factor)

        # ~ EDGE IMPORTANCES
        self.lay_act_importance = ActivationEmbedding(activation=self.importance_activation)
        self.lay_concat_alphas = LazyConcatenate(axis=-1)

        self.lay_pool_edges_in = PoolingLocalEdges(pooling_method='mean', pooling_index=0)
        self.lay_pool_edges_out = PoolingLocalEdges(pooling_method='mean', pooling_index=1)
        self.lay_average = LazyAverage()

        # ~ NODE IMPORTANCES
        self.node_importance_units = importance_units + [self.importance_channels]
        self.node_importance_acts = ['kgcnn>leaky_relu' for _ in importance_units] + ['linear']
        self.node_importance_layers = []
        for u, act in zip(self.node_importance_units, self.node_importance_acts):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=use_bias
            )
            self.node_importance_layers.append(lay)

        # ~ OUTPUT / MLP TAIL END
        self.lay_pool_out = PoolingNodes(pooling_method=self.final_pooling)
        self.lay_concat_out = LazyConcatenate(axis=-1)
        self.lay_final_dropout = DropoutEmbedding(rate=self.final_dropout_rate)

        self.final_acts = ['kgcnn>leaky_relu' for _ in self.final_units]
        self.final_acts[-1] = self.final_activation
        self.final_biases = [True for _ in self.final_units]
        self.final_biases[-1] = True
        self.final_layers = []
        for u, act, bias in zip(self.final_units, self.final_acts, self.final_biases):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=use_bias
            )
            self.final_layers.append(lay)

        # ~ EXPLANATION ONLY TRAIN STEP
        self.bce_loss = ks.losses.BinaryCrossentropy()
        self.compiled_classification_loss = compile_utils.LossesContainer(self.bce_loss)

        self.mse_loss = ks.losses.MeanSquaredError()
        self.mae_loss = ks.losses.MeanAbsoluteError()
        self.compiled_regression_loss = compile_utils.LossesContainer(mae)

        # If regression_limits have been supplied, we interprete this as the intent to perform explanation
        # co-training for a regression dataset.
        # So the content of this if condition makes sure to perform the necessary pre-processing steps
        # for this case.
        if self.regression_limits is not None:
            # first of all we do some simple assertions to handle some common error cases
            assert regression_reference is not None, ('You have to supply a "regression_reference" value for '
                                                      'explanation co-training!')

            # This is the first and simpler case for regression explanation co-training: In this case the
            # regression reference value is only a single value. In that case, there is only one target
            # value that is supposed to be regressed. The alternative would be that it is a list in which
            # case it would have to have as many elements as target values to be predicted.
            # However in this case we convert it into a list as well to be able to treat everything from
            # this point on as the multi-value case guaranteed.
            if isinstance(regression_reference, (int, float)):
                self.regression_reference = [regression_reference]

            num_references = len(self.regression_reference)
            num_limits = len(self.regression_limits)
            assert num_references * 2 == importance_channels, (
                f'for explanation co-training, the number of regression_references (currently {num_references}) has '
                f'to be exactly half the number of importance channels (currently {importance_channels})!'
            )
            assert num_references == final_units[-1], (
                f'For explanation co-training, the number of regression_references (currently {num_references}) has '
                f'to be exactly the same as the final unit count in the MLP tail end (currently {final_units[-1]})'
            )
            assert num_references == num_limits, (
                f'For explanation co-training, the number of regression_references (currently {num_references}) has '
                f'to be exactly the same as the number of regression_limits intervals (currently {num_limits})'
            )

    def get_config(self):
        config = super(Megan, self).get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "use_edge_features": self.use_edge_features,
            "importance_units": self.importance_units,
            "importance_channels": self.importance_channels,
            "importance_activation": self.importance_activation,
            "importance_dropout_rate": self.importance_dropout_rate,
            "importance_factor": self.importance_factor,
            "importance_multiplier": self.importance_multiplier,
            "sparsity_factor": self.sparsity_factor,
            "concat_heads": self.concat_heads,
            "final_units": self.final_units,
            "final_dropout_rate": self.final_dropout_rate,
            "final_activation": self.final_activation,
            "final_pooling": self.final_pooling,
            "regression_limits": self.regression_limits,
            "regression_reference": self.regression_reference,
            "return_importances": self.return_importances
        })

        return config
    @property
    def doing_regression(self) -> bool:
        return self.regression_limits is not None

    def call(self,
             inputs,
             training: bool = False,
             return_importances: bool = False,
             node_importances_mask: t.Optional[tf.RaggedTensor] = None,
             **kwargs):
        """
        Forward pass of the model.

        **Shape Explanations:** All shapes in brackets [] are ragged dimensions!

        - V: Num nodes in the graph
        - E: Num edges in the graph
        - N: Num feature values per node
        - M: NUm feature values per edge
        - H: Num feature values per graph
        - B: Num graphs in a batch
        - K: Num importance (explanation) channels configured in the constructor
        """

        # 17.11.2022
        # Added support for global graph attributes. If the corresponding flag is set in the constructor of
        # the model then it is expected that the input tuple consists of 4 elements instead of the usual
        # 3 elements, where the fourth element is the vector of the graph attributes.
        # We can't use these graph attributes right away, but later on we will simply append them to the
        # vector which enters the MLP tail end.

        # node_input: ([B], [V], N)
        # edge_input: ([B], [E], M)
        # edge_index_input: ([B], [E], 2)
        # graph_input: ([B], H)
        if self.use_graph_attributes:
            node_input, edge_input, edge_index_input, graph_input = inputs
        else:
            node_input, edge_input, edge_index_input = inputs
            graph_input = None

        # First of all we apply all the graph convolutional / attention layers. Each of those layers outputs
        # the attention logits alpha additional to the node embeddings. We collect all the attention logits
        # in a list so that we can later sum them all up.
        alphas = []
        x = node_input
        for lay in self.attention_layers:
            # x: ([batch], [N], F)
            # alpha: ([batch], [M], K, 1)
            x, alpha = lay([x, edge_input, edge_index_input])
            if training:
                x = self.lay_dropout(x, training=training)

            alphas.append(alpha)

        # We sum up all the individual layers attention logit tensors and the edge importances are directly
        # calculated by applying a sigmoid on that sum.
        alphas = self.lay_concat_alphas(alphas)
        edge_importances = tf.reduce_sum(alphas, axis=-1, keepdims=False)
        edge_importances = self.lay_act_importance(edge_importances)

        # Part of the final node importance tensor is actually the pooled edge importances, so that is what
        # we are doing here. The caveat here is that we assume undirected edges as two directed edges in
        # opposing direction. To now achieve a symmetric pooling of these edges we have to pool in both
        # directions and then use the average of both.
        pooled_edges_in = self.lay_pool_edges_in([node_input, edge_importances, edge_index_input])
        pooled_edges_out = self.lay_pool_edges_out([node_input, edge_importances, edge_index_input])
        pooled_edges = self.lay_average([pooled_edges_out, pooled_edges_in])

        node_importances_tilde = x
        for lay in self.node_importance_layers:
            node_importances_tilde = lay(node_importances_tilde)

        node_importances_tilde = self.lay_act_importance(node_importances_tilde)

        node_importances = node_importances_tilde * pooled_edges
        self.lay_sparsity(node_importances)

        # ~ Applying the node importance mask
        # "node_importances_mask" is supposed to be a ragged tensor of the exact same dimensions as the
        # node importances, containing binary values 0 or 1, which are then used as a multiplicative mask
        # to modify the actual node importances before the global pooling step.
        # The main use case of this feature is to completely mask out certain channels to see how that
        # the missing channels (linked to a certain explanation / interpretation) affect the outcome of
        # the MLP tail end.
        if node_importances_mask is not None:
            node_importances_mask = tf.cast(node_importances_mask, tf.float32)
            node_importances = node_importances * node_importances_mask

        # Here we apply the global pooling. It is important to note that we do K separate pooling operations
        # were each time we use the same node embeddings x but a different slice of the node importances as
        # the weights! We concatenate all the individual results in the end.
        outs = []
        n = self.final_units[-1]
        for k in range(self.importance_channels):
            node_importance_slice = tf.expand_dims(node_importances[:, :, k], axis=-1)
            masked_embeddings = x * node_importance_slice

            # 26.03.2023
            # Optionally, if given we apply an additional non-linear transformation in the form of an
            # additional layer on each of the masked node embeddings separately.
            if self.importance_transformations is not None:
                lay_transform = self.importance_transformations[k]
                masked_embeddings = lay_transform(masked_embeddings)

            out = self.lay_pool_out(masked_embeddings)
            # out = self.lay_pool_out(x[:, :, k*n:(k+1)*n] * node_importance_slice)

            outs.append(out)

        # out: ([B], N*K²)
        out = self.lay_concat_out(outs)

        # At this point, after the global pooling of the node embeddings, we can append the global graph
        # attributes, should those exist
        if self.use_graph_attributes:
            out = self.lay_concat_out([out, graph_input])

        # Now "out" is a graph embedding vector of known dimension so we can simply apply the normal dense
        # mlp to get the final output value.
        num_final_layers = len(self.final_layers)
        for c, lay in enumerate(self.final_layers):
            out = lay(out)
            if training and c < num_final_layers - 2:
                out = self.lay_final_dropout(out, training=training)

        if self.doing_regression:
            reference = tf.ones_like(out) * tf.constant(self.regression_reference, dtype=tf.float32)
            out = out + reference

        # Usually, the node and edge importance tensors would be direct outputs of the model as well, but
        # we need the option to just return the output alone to be compatible with the standard model
        # evaluation pipeline already implemented in the library.
        if self.return_importances or return_importances:
            return out, node_importances, edge_importances
        else:
            return out

    def regression_augmentation(self,
                                out_true):
        samples = []
        masks = []
        for i, (regression_reference, regression_limits) in enumerate(zip(self.regression_reference,
                                                                          self.regression_limits)):

            regression_width = abs(regression_limits[1] - regression_limits[0])
            values = tf.expand_dims(out_true[:, i], axis=-1)
            center_distances = tf.abs(values - regression_reference)
            center_distances = (center_distances * self.importance_multiplier) / (0.5 * regression_width)

            # So we need two things: a "samples" tensor and a "mask" tensor. We are going to use the samples
            # tensor as the actual ground truth which acts as the regression target during the explanation
            # train step. The binary values of the mask will determine at which positions a loss should
            # actually be calculated for both of the channels

            # The "lower" part is all the samples which have a target value below the reference value.
            lo_mask = tf.where(values < regression_reference, 1.0, 0.0)
            # The "higher" part is all the samples above reference
            hi_mask = tf.where(values > regression_reference, 1.0, 0.0)

            samples += [center_distances, center_distances]
            masks += [lo_mask, hi_mask]

        return (
            tf.concat(samples, axis=-1),
            tf.concat(masks, axis=-1)
        )

    def train_step_explanation(self, x, y,
                               update_weights: bool = True):

        if self.return_importances:
            out_true, _, _ = y
        else:
            out_true = y

        exp_loss = 0
        with tf.GradientTape() as tape:

            y_pred = self(x, training=True, return_importances=True)
            out_pred, ni_pred, ei_pred = y_pred

            # ~ explanation loss
            # First of all we need to assemble the approximated model output, which is simply calculated
            # by applying a global pooling operation on the corresponding slice of the node importances.
            # So for each slice (each importance channel) we get a single value, which we then
            # concatenate into an output vector with K dimensions.
            outs = []
            for k in range(self.importance_channels):
                node_importances_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)
                out = self.lay_pool_out(node_importances_slice)

                outs.append(out)

            # outs: ([batch], K)
            outs = self.lay_concat_out(outs)

            if self.doing_regression:
                out_true, mask = self.regression_augmentation(out_true)
                out_pred = outs
                exp_loss = self.importance_channels * self.compiled_regression_loss(out_true * mask,
                                                                                    out_pred * mask)

            else:
                # out_pred = ks.backend.sigmoid(outs)
                out_pred = shifted_sigmoid(outs, shift=self.importance_multiplier, multiplier=2)
                exp_loss = self.compiled_classification_loss(out_true, out_pred * out_true)

            exp_loss *= self.importance_factor

        # Compute gradients
        trainable_vars = self.trainable_variables
        exp_gradients = tape.gradient(exp_loss, trainable_vars)

        # Update weights
        if update_weights:
            self.optimizer.apply_gradients(zip(exp_gradients, trainable_vars))

        return {'exp_loss': exp_loss}

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # The "train_step_explanation" method will execute the entire explanation train step once. This
        # mainly involves creating an approximate solution of the original regression / classification
        # problem using ONLY the node importances of the different channels and then performing one
        # complete weight update based on the corresponding loss.
        if self.importance_factor != 0 and self.separate_explanation_step:
            exp_metrics = self.train_step_explanation(x, y)
        else:
            exp_metrics = {}

        exp_loss = 0
        with tf.GradientTape() as tape:

            out_true, ni_true, ei_true = y
            out_pred, ni_pred, ei_pred = self(x, training=True, return_importances=True)
            loss = self.compiled_loss(
                [out_true, ni_true, ei_true],
                [out_pred, ni_pred, ei_pred],
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            if self.importance_factor != 0 and not self.separate_explanation_step:
                # ~ explanation loss
                # First of all we need to assemble the approximated model output, which is simply calculated
                # by applying a global pooling operation on the corresponding slice of the node importances.
                # So for each slice (each importance channel) we get a single value, which we then
                # concatenate into an output vector with K dimensions.
                outs = []
                for k in range(self.importance_channels):
                    node_importances_slice = tf.expand_dims(ni_pred[:, :, k], axis=-1)
                    out = self.lay_pool_out(node_importances_slice)

                    outs.append(out)

                # outs: ([batch], K)
                outs = self.lay_concat_out(outs)

                if self.doing_regression:
                    _out_true, mask = self.regression_augmentation(out_true)
                    _out_pred = outs
                    exp_loss = self.compiled_regression_loss(_out_true * mask,
                                                             _out_pred * mask)

                else:
                    # out_pred = ks.backend.sigmoid(outs)
                    #_out_pred = shifted_sigmoid(outs, multiplier=self.importance_multiplier)
                    _out_pred = shifted_sigmoid(
                        outs,
                        shift=self.importance_multiplier,
                        multiplier=1
                    ) * out_true
                    exp_loss = self.compiled_classification_loss(out_true, _out_pred)

                exp_loss *= self.importance_factor
                exp_metrics['exp_loss'] = exp_loss
                loss += exp_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(
            y,
            out_pred if not self.return_importances else [out_pred, ni_pred, ei_pred],
            sample_weight=sample_weight
        )

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            **{m.name: m.result() for m in self.metrics},
            **exp_metrics
        }


class GnesGrad(ks.models.Model):

    def __init__(self,
                 units: t.List[int],
                 batch_size: int,
                 num_outputs: int = 1,
                 pooling_method: str = 'mean'):
        super(GnesGrad, self).__init__()
        self.batch_size = batch_size

        self.conv_layers = []
        for k in units:
            #lay = AttentionHeadGAT(units=k)
            lay = GCN(units=k)
            self.conv_layers.append(lay)

        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        self.lay_dense = DenseEmbedding(units=num_outputs)

    def call(self, inputs, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        node_input, edge_input, edge_index_input = inputs
        edge_input = tf.cast(edge_input, dtype=tf.float32)
        x = node_input

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(node_input)
            tape.watch(edge_input)

            activations = []
            for lay in self.conv_layers:
                x = lay([x, edge_input, edge_index_input])
                activations.append(x)

            out = self.lay_pooling(x)
            out = self.lay_dense(out)

            out_list = [out[i, 0] for i in range(batch_size)]

        node_gradients = []
        edge_gradients = []
        for i in range(batch_size):
            out_value = out_list[i]

            node_gradient = tape.gradient(out_value, node_input)[i]
            node_gradients.append(node_gradient)

            edge_gradient = tape.gradient(out_value, edge_input)[i]
            edge_gradients.append(edge_gradient)

        node_gradients = tf.ragged.stack(node_gradients, axis=0)
        edge_gradients = tf.ragged.stack(edge_gradients, axis=0)

        node_importances = tf.reduce_mean(node_gradients, axis=-1, keepdims=True)
        node_importances = tf.concat([
            tf.where(node_importances > 0, node_importances, 0.),
            tf.where(node_importances < 0, -node_importances, 0.),
        ], axis=-1)

        edge_importances = tf.reduce_mean(edge_gradients, axis=-1, keepdims=True)
        edge_importances = tf.concat([
            tf.where(edge_importances < 0, -edge_importances, 0.),
            tf.where(edge_importances > 0, edge_importances, 0.),
        ], axis=-1)

        return out, node_importances, edge_importances


class GnesGradCam(ks.models.Model):

    def __init__(self,
                 units: t.List[int],
                 batch_size: int,
                 num_outputs: int = 1,
                 pooling_method: str = 'mean'):
        super(GnesGradCam, self).__init__()
        self.batch_size = batch_size

        self.conv_layers = []
        for k in units:
            #lay = AttentionHeadGAT(units=k)
            lay = GCN(units=k)
            self.conv_layers.append(lay)

        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)
        self.lay_dense = DenseEmbedding(units=num_outputs)
        self.lay_concat = LazyConcatenate(axis=0)

    def call(self, inputs, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        node_input, edge_input, edge_index_input = inputs
        x = node_input

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(node_input)
            tape.watch(edge_input)

            activations = []
            for lay in self.conv_layers:
                x = lay([x, edge_input, edge_index_input])
                activations.append(x)

            out = self.lay_pooling(x)
            out = self.lay_dense(out)

            out_list = [out[i, 0] for i in range(batch_size)]

        node_importances_list: t.List[t.List[tf.Tensor]] = []
        edge_importances_list: t.List[t.List[tf.Tensor]] = []
        for l, activation in enumerate(activations):
            node_gradients = []
            edge_gradients = []
            for i in range(batch_size):
                out_value = out_list[i]

                node_gradient = tape.gradient(out_value, activation)
                node_gradient = tf.expand_dims(node_gradient, axis=0)
                node_gradients.append(node_gradient)

            node_gradients = tf.concat(node_gradients, axis=0)
            node_gradients = tf.reduce_sum(node_gradients, axis=1)

            node_alpha = tf.reduce_mean(node_gradients, axis=1, keepdims=True)
            node_importances = activation * node_alpha
            node_importances = tf.reduce_sum(node_importances, axis=-1, keepdims=True)
            node_importances_list.append(node_importances)

        node_importances = tf.concat(node_importances_list, axis=-1)
        node_importances = tf.reduce_mean(node_importances, axis=-1, keepdims=True)
        node_importances = tf.concat([
            tf.where(node_importances < 0, -node_importances, 0.),
            tf.where(node_importances > 0, node_importances, 0.),
        ], axis=-1)

        edge_gradients = []
        for i in range(batch_size):
            out_value = out_list[i]

            edge_gradient = tape.gradient(out_value, edge_input)[i]
            edge_gradients.append(edge_gradient)

        edge_gradients = tf.ragged.stack(edge_gradients, axis=0)

        edge_importances = tf.reduce_mean(edge_gradients, axis=-1, keepdims=True)
        edge_importances = tf.concat([
            tf.where(edge_importances < 0, -edge_importances, 0.),
            tf.where(edge_importances > 0, edge_importances, 0.),
        ], axis=-1)

        return out, node_importances, edge_importances

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class AbstractGradientModel(ks.models.Model):
    """
    This is an abstract base class for models which will provide their gradient information for further
    processing. This means that during the inference call, these kinds of models have to maintain their
    own gradient tape to record the full inference process and then use this to provide gradients of the
    input and intermediate node / edge embeddings with respect to the output.
    """

    def call_with_gradients(self, inputs, training=True):
        """
        **How the list of gradients should look like**

        Assuming output dimension H, number of nodes V, node features N and batch size B, as well as the
        number of convolutional layers L in the network.

        The list should consist of L+1 elements, where the first element is always the gradients with respect
        to the input and after that the gradients of the intermediate node/edge embeddings.

        Each element should be a ragged tensor with the dimension ([B], [V], N, H)

        :return: This has to be a tuple of four values:
            - The main output prediction "y" of the model
            - A list containing the gradients of the input as well as the intermediate node(!) embeddings
              w.r.t to the output "y".
            - A list containing the gradients of the input as well as the intermediate edge(!) embeddings
              w.r.t to the output "y".
            - The gradient tape used to create the previously mentioned gradients
        """
        raise NotImplementedError()

    def call(self, inputs, training=True, *args, **kwargs):
        y, node_gradients, edge_gradients, tape = self.call_with_gradients(inputs, training, *args, **kwargs)
        return y

    def calculate_gradient_info(self,
                                outs: t.List[t.List[tf.Tensor]],
                                activations: t.List[tf.Tensor],
                                tape: tf.GradientTape,
                                batch_size):
        """
        This method is a boilerplate implementation that will calculate the gradient info list, given a list
        of model outputs, a list of layer activation tensors, the gradient tape and the batch size.

        Assuming the following variables:
        - B: batch dimension
        - O: output dimension
        - L: number of convolutional layers / number of intermediate embeddings w.r.t which the gradients
            are to be calculated
        - V: Node or edge shape
        - F: Feature shape

        :param outs: This is supposed to be a list of lists of zero dimensional tensors. Generally, the
            output of a regression / classification network will be a vector of shape ([B], O). This list
            has to basically be this tensor only in list format. This list should be created by applying
            the corresponding indexing slices to the output vector to split it into the nested format of
            zero dimensional tensor objects. Note that this indexing operation has to be done INSIDE the
            gradient tape context, otherwise the calculation of the gradients will not work!
        :param activations: This is a list with L elements, where each element is an intermediate node /
            edge embedding vector from the call process of the network. In this method one gradient vector
            will be created for each of the outputs w.r.t to each of these embeddings. The resulting
            gradient tensors will have the same shape as the embeddings!
        :param tape: The GradientTape which was used during the call process.
        :param batch_size: The batch size B that was used to obtain the outputs and the embeddings. This has
            to be exactly correct!
        :return: A list of tuples with the length L. Each tuple contains two elements. The first one is the
            exact same embedding tensor that was passed to the method in the "activations" parameter.
            This tensor has the shape ([B], [V], F). The second element is the corresponding gradient
            tensor that was created for that very embedding with the shape ([B], [V], F, O). Note that the
            gradient has an additional dimension for the output shape. This is because gradients are
            created for every output.
        """
        gradient_info = []
        for activation in activations:
            gradients = []
            for b in range(batch_size):

                # In this inner loop here we manage the fact that the network may have multiple outputs
                # O != 1, so we iterate over every one of the outputs for the current element of the batch
                # add them to "gradient_stack" and then concat them in the end. The important part here is
                # that we want to stack them such that the resulting shape is ([B], [V], O, N) and not
                # ([B], [V], N, O) as you would intuitively do it by concat over the last dimension!
                gradient_stack = []
                for out in outs[b]:
                    # What we get out of this operation is the following shape ([B], [V], N) but technically
                    # the first dimension is pointless, because we are calculating the gradients of only one
                    # specific output value in the batch with respect to every element in the batch. Only
                    # one element in this dimension will be != zero. But actually we need to do it this way
                    # so that the ragged concat operation further down the line will work. If we already only
                    # select the appropriate element here, we get a serious issue with the stacking of ragged
                    # dimensions later.
                    gradient = tape.gradient(out, activation)
                    gradient = tf.expand_dims(gradient, axis=-2)
                    gradient_stack.append(gradient)

                # gradient_stack will be a vector of shape ([B], [V], O, N, 1). The last dimension we have
                # expanded here because in the following we need to stack this over the batch dimension
                # again.
                gradient_stack = tf.concat(gradient_stack, axis=-2)
                gradient_stack = tf.expand_dims(gradient_stack, axis=-1)

                gradients.append(gradient_stack)

            # results in a tensor of shape ([B], [V], O, N, [B])
            gradients = tf.concat(gradients, axis=-1)
            # So this is the point AFTER the concat operation where we can savely get rid of the one
            # batch dimension we have uselessly kept before.
            # results in fully populated gradient vector of shape ([B], [V], N) exactly as we wanted
            gradients = tf.reduce_sum(gradients, axis=-1)
            gradient_info.append((activation, gradients))

        return gradient_info


class GcnGradientModel(AbstractGradientModel, GNNInterface):

    def __init__(self,
                 batch_size: int,
                 units: t.List[int],
                 final_units: t.List[int],
                 layer_cb: t.Callable = lambda units: GCN(units=units),
                 pooling_method: str = 'mean',
                 final_activation: str = 'linear'):
        AbstractGradientModel.__init__(self)
        GNNInterface.__init__(self)

        self.batch_size = batch_size
        self.units = units
        self.num_outputs = final_units[-1]

        self.conv_layers = []
        for k in self.units:
            lay = layer_cb(k)
            self.conv_layers.append(lay)

        self.lay_pooling = PoolingNodes(pooling_method=pooling_method)

        self.final_units = final_units
        self.final_acts = ['kgcnn>leaky_relu' for _ in final_units]
        self.final_acts[-1] = final_activation
        self.final_layers = []
        for k, act in zip(self.final_units, self.final_acts):
            lay = DenseEmbedding(units=k, activation=act)
            self.final_layers.append(lay)

    def call_with_gradients(self, inputs, training=True, create_gradients=True, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        node_input, edge_input, edge_index_input = inputs
        x = node_input

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(node_input)
            tape.watch(edge_input)

            edge_activations = [edge_input]
            node_activations = [x]
            for lay in self.conv_layers:
                x = lay([x, edge_input, edge_index_input])
                node_activations.append(x)

            out = self.lay_pooling(x)

            for lay in self.final_layers:
                out = lay(out)

            outs = [out[b, :] for b in range(batch_size)]
            outs_multi = [[out[b, o] for o in range(self.num_outputs)] for b in range(batch_size)]

        if create_gradients:
            edge_gradient_info = self.calculate_gradient_info(outs_multi, edge_activations, tape, batch_size)
            node_gradient_info = self.calculate_gradient_info(outs_multi, node_activations, tape, batch_size)
        else:
            node_gradient_info = []
            edge_gradient_info = []

        return out, node_gradient_info, edge_gradient_info, tape

    def call(self,
             inputs,
             training=True,
             batch_size=None,
             create_gradients=True,
             return_gradients=False):
        y, node_info, edge_info, tape = self.call_with_gradients(
            inputs,
            training=training,
            batch_size=batch_size,
            create_gradients=create_gradients
        )

        if return_gradients:
            return y, node_info, edge_info
        else:
            return y


def grad_importances(gradient_info: t.List[t.Tuple[tf.Tensor, tf.Tensor]],
                     use_relu: bool = False,
                     use_absolute: bool = False,
                     keepdims: bool = False):
    # shapes as we get them:
    # gradients: ([B], [V], O, F)
    # activation: ([B], [V], F)

    activation, gradients = gradient_info[0]
    importances = tf.reduce_sum(
        # We need to expand the activations here in the second last dimension to match the additional
        # output shape O of the gradients.
        gradients * tf.expand_dims(activation, axis=-2),
        axis=-1,
        keepdims=keepdims
    )
    if use_relu:
        importances = ks.backend.relu(importances)
    if use_absolute:
        importances = tf.abs(importances)

    return importances


def grad_cam_importances(gradient_info: t.List[t.Tuple[tf.Tensor, tf.Tensor]],
                         use_average: bool = False,
                         use_relu: bool = False,
                         use_absolute: bool = False,
                         averaging_depth: int = 2,
                         keepdims: bool = False):
    importances = []
    for activation, gradients in gradient_info:
        # shapes as we get them:
        # gradients: ([B], [V], O, F)
        # activation: ([B], [V], F)

        alpha = tf.reduce_mean(gradients, axis=1, keepdims=True)
        local_importances = tf.reduce_sum(
            # We need to expand the activations here in the second last dimension to match the additional
            # output shape O of the gradients.
            alpha * tf.expand_dims(activation, axis=-2),
            axis=-1,
            keepdims=keepdims
        )

        if use_relu:
            local_importances = ks.backend.relu(local_importances)
        if use_absolute:
            local_importances = tf.abs(local_importances)

        importances.append(local_importances)

    if use_average:
        importances = tf.concat(importances[-averaging_depth:], axis=-1)
    else:
        importances = importances[-1]

    importances = tf.reduce_mean(importances, axis=-1, keepdims=True)
    return importances


def gnnx_importances(model,
                     x,
                     y,
                     epochs: int = 100,
                     learning_rate: float = 0.01,
                     node_sparsity_factor: float = 1.0,
                     edge_sparsity_factor: float = 1.0,
                     model_kwargs: dict = {},
                     logger: logging.Logger = NULL_LOGGER,
                     log_step: int = 10):
    logger.info('creating explanations with gnn explainer')
    start_time = time.time()

    optimizer = ks.optimizers.Nadam(learning_rate=learning_rate)

    node_input = x[0]
    edge_input = x[1]
    edge_index_input = x[2]

    node_mask_ragged = tf.reduce_mean(tf.ones_like(node_input), axis=-1, keepdims=True)
    node_mask_variables = tf.Variable(node_mask_ragged.flat_values, trainable=True, dtype=tf.float64)

    edge_mask_ragged = tf.reduce_mean(tf.ones_like(edge_input), axis=-1, keepdims=True)
    edge_mask_variables = tf.Variable(edge_mask_ragged.flat_values, trainable=True, dtype=tf.float64)

    for epoch in range(epochs):

        with tf.GradientTape() as tape:
            node_mask = tf.RaggedTensor.from_nested_row_splits(
                node_mask_variables,
                nested_row_splits=node_mask_ragged.nested_row_splits
            )
            node_masked = node_input * node_mask

            edge_mask = tf.RaggedTensor.from_nested_row_splits(
                edge_mask_variables,
                nested_row_splits=edge_mask_ragged.nested_row_splits
            )
            edge_masked = edge_input * edge_mask

            out = model([
                node_masked,
                edge_masked,
                edge_index_input,
                *x[3:]
            ], **model_kwargs)

            loss = tf.cast(tf.reduce_mean(tf.square(y - out)), dtype=tf.float64)
            loss += node_sparsity_factor * tf.reduce_mean(tf.abs(node_mask))
            loss += edge_sparsity_factor * tf.reduce_mean(tf.abs(edge_mask))

        trainable_vars = [node_mask_variables, edge_mask_variables]
        gradients = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        if epoch % log_step == 0:
            logger.info(f' ({epoch}/{epochs})'
                        f' - loss: {loss}'
                        f' - elapsed time: {time.time() - start_time:.2f}')

    return (
        node_mask,
        edge_mask
    )


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

