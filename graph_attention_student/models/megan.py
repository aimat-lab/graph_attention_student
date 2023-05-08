import typing as t

import tensorflow as tf
import tensorflow.keras as ks
import visual_graph_datasets.typing as tv
from tensorflow.python.keras.engine import compile_utils
from kgcnn.data.utils import ragged_tensor_from_nested_numpy
from kgcnn.layers.modules import GraphBaseLayer
from kgcnn.layers.modules import LazyConcatenate, LazyAverage
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, DropoutEmbedding
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.pooling import PoolingWeightedNodes

from graph_attention_student.data import process_graph_dataset
from graph_attention_student.layers import ExplanationSparsityRegularization
from graph_attention_student.layers import ExplanationGiniRegularization
from graph_attention_student.layers import MultiHeadGATV2Layer
from graph_attention_student.training import mae
from graph_attention_student.training import bce
from graph_attention_student.training import shifted_sigmoid


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
                 gini_factor: float = 0.0,
                 concat_heads: bool = True,
                 separate_explanation_step: bool = False,
                 # mlp tail end related arguments
                 final_units: t.List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 final_pooling: str = 'sum',
                 regression_limits: t.Optional[t.Tuple[float, float]] = None,
                 regression_weights: t.Optional[t.Tuple[float, float]] = None,
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
        self.gini_factor = gini_factor
        self.concat_heads = concat_heads
        self.final_units = final_units
        self.final_dropout_rate = final_dropout_rate
        self.final_activation = final_activation
        self.final_pooling = final_pooling
        self.regression_limits = regression_limits
        self.regression_weights = regression_weights
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
        self.lay_gini = ExplanationGiniRegularization(
            factor=self.gini_factor,
            num_channels=importance_channels
        )

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
        self.compiled_classification_loss = compile_utils.LossesContainer(bce)

        self.mse_loss = ks.losses.MeanSquaredError()
        self.mae_loss = ks.losses.MeanAbsoluteError()
        self.compiled_regression_loss = compile_utils.LossesContainer(mae)

        # TODO: Clean up this mess
        # If regression_limits have been supplied, we interprete this as the intent to perform explanation
        # co-training for a regression dataset.
        # So the content of this if condition makes sure to perform the necessary pre-processing steps
        # for this case.
        if self.regression_reference is not None:

            # This is the first and simpler case for regression explanation co-training: In this case the
            # regression reference value is only a single value. In that case, there is only one target
            # value that is supposed to be regressed. The alternative would be that it is a list in which
            # case it would have to have as many elements as target values to be predicted.
            # However in this case we convert it into a list as well to be able to treat everything from
            # this point on as the multi-value case guaranteed.
            if isinstance(regression_reference, (int, float)):
                self.regression_reference = [regression_reference]

            num_references = len(self.regression_reference)

            if self.regression_weights is not None:
                num_values = len(self.regression_weights)
            elif self.regression_limits is not None:
                num_values = len(self.regression_limits)
            else:
                raise AssertionError(f'You have supplied a non-null value for regression_reference: '
                                     f'{self.regression_reference}. That means you need to either supply '
                                     f'a valid value for regression_limits or regression_weights as well')

            assert num_references * 2 == importance_channels, (
                f'for explanation co-training, the number of regression_references (currently {num_references}) has '
                f'to be exactly half the number of importance channels (currently {importance_channels})!'
            )
            assert num_references == final_units[-1], (
                f'For explanation co-training, the number of regression_references (currently {num_references}) has '
                f'to be exactly the same as the final unit count in the MLP tail end (currently {final_units[-1]})'
            )
            assert num_references == num_values, (
                f'For explanation co-training, the number of regression_references (currently {num_references}) has '
                f'to be exactly the same as the number of regression_limits intervals (currently {num_values})'
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
            "regression_weights": self.regression_weights,
            "regression_reference": self.regression_reference,
            "return_importances": self.return_importances,
            "use_graph_attributes": self.use_graph_attributes,
        })

        return config

    # ~ Properties

    @property
    def doing_regression(self) -> bool:
        return (self.regression_limits is not None) or (self.regression_weights is not None)

    def doing_regression_weights(self) -> bool:
        return self.regression_weights is not None

    # ~ Forward Pass Implementation

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
        if self.sparsity_factor > 0:
            self.lay_sparsity(node_importances)
        if self.gini_factor > 0:
            self.lay_gini(node_importances)

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

        for i, regression_reference in enumerate(self.regression_reference):
            values = tf.expand_dims(out_true[:, i], axis=-1)
            center_distances = tf.abs(values - regression_reference)

            if self.doing_regression_weights:
                regression_weights = self.regression_weights[i]
                center_distances = tf.where(
                    values < regression_reference,
                    center_distances * (self.importance_multiplier * regression_weights[0]),
                    center_distances * (self.importance_multiplier * regression_weights[1]),
                )

            else:
                regression_limits = self.regression_limits[i]
                regression_width = abs(regression_limits[1] - regression_limits[0])
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

    # -- Implements "PredictGraphMixin"
    # These following method implementations are required to

    def predict_graphs(self, graph_list: t.List[tv.GraphDict]):
        indices = list(range(len(graph_list)))
        x, _, _, _ = process_graph_dataset(
            dataset=graph_list,
            train_indices=indices,
            test_indices=indices,
            use_importances=False,
            use_graph_attributes=False,
        )

        return list(zip(*[v.numpy() for v in self(x)]))

    def predict_graph(self, graph: tv.GraphDict):
        return self.predict_graphs([graph])[0]



class FidelityMegan(Megan):

    def __init__(self,
                 *args,
                 fidelity_factor: float = 0.0,
                 fidelity_funcs: t.List[t.Callable] = [],
                 final_activation: str = 'linear',
                 **kwargs
                 ):
        super(FidelityMegan, self).__init__(
            *args,
            final_activation='linear',
            **kwargs
        )
        self.fidelity_factor = fidelity_factor
        self.fidelity_funcs = fidelity_funcs

        self.lay_final_activation = ActivationEmbedding(activation=final_activation)
        self.bias = tf.Variable(initial_value=tf.zeros(shape=(self.final_units[-1], )))

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
        if self.sparsity_factor > 0:
            self.lay_sparsity(node_importances)
        if self.gini_factor > 0:
            self.lay_gini(node_importances)

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

            num_final_layers = len(self.final_layers)
            for c, lay in enumerate(self.final_layers):
                out = lay(out)
                if training and c < num_final_layers - 2:
                    out = self.lay_final_dropout(out, training=training)

            outs.append(out)

        # out: ([B], N*K²)
        # outs = tf.concat(outs, axis=-1)
        # out = self.lay_final_dense(outs)
        outs = tf.concat([tf.expand_dims(out, axis=-1) for out in outs], axis=-1)
        out = tf.reduce_sum(outs, axis=-1) + self.bias
        out = self.lay_final_activation(out)

        # Usually, the node and edge importance tensors would be direct outputs of the model as well, but
        # we need the option to just return the output alone to be compatible with the standard model
        # evaluation pipeline already implemented in the library.
        if self.return_importances or return_importances:
            return out, node_importances, edge_importances
        else:
            return out

    def train_step_fidelity(self, data, out_pred, ni_pred, ei_pred):

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        loss = 0
        with tf.GradientTape() as tape:
            ones = tf.reduce_mean(tf.ones_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)
            zeros = tf.reduce_mean(tf.zeros_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)

            for channel_index, func in enumerate(self.fidelity_funcs):
                mask = [zeros if i == channel_index else ones for i in range(self.importance_channels)]
                mask = tf.concat(mask, axis=-1)
                out_mod, _, _ = self(
                    x,
                    training=True,
                    return_importances=True,
                    node_importances_mask=mask,
                )
                diff = func(out_pred, out_mod)
                loss += diff

            loss *= self.fidelity_factor

        trainable_vars = self.trainable_variables
        #print(len(trainable_vars))
        final_vars = [weight.name for lay in self.final_layers for weight in lay.weights]
        #print(final_vars)
        trainable_vars = [var for var in trainable_vars if var.name not in final_vars]
        #print(len(trainable_vars))

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return loss

    def train_step(self, data):

        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        exp_metrics = {'exp_loss': 0}
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

            if self.fidelity_factor > 0:
                ones = tf.reduce_mean(tf.ones_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)
                zeros = tf.reduce_mean(tf.zeros_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)

                for channel_index, func in enumerate(self.fidelity_funcs):
                    mask = [zeros if i == channel_index else ones for i in range(self.importance_channels)]
                    mask = tf.concat(mask, axis=-1)
                    out_mod, _, _ = self(
                        x,
                        training=True,
                        return_importances=True,
                        node_importances_mask=mask,
                    )
                    diff = func(out_pred, out_mod)
                    exp_loss += diff

                exp_loss *= self.fidelity_factor
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
