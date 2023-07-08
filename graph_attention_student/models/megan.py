import typing as t

import numpy as np
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
from graph_attention_student.data import tensors_from_graphs
from graph_attention_student.layers import ExplanationSparsityRegularization
from graph_attention_student.layers import ExplanationGiniRegularization
from graph_attention_student.layers import MultiHeadGATV2Layer
from graph_attention_student.training import mae
from graph_attention_student.training import bce
from graph_attention_student.training import shifted_sigmoid


class MockMegan:

    def __init__(self,
                 importance_channels: int,
                 final_units: t.List[int],
                 *args,
                 **kwargs,
                 ):
        self.importance_channels = importance_channels
        self.final_units = final_units
        self.num_targets = final_units[-1]

    def __call__(self, x, *args, **kwargs):
        node_input, edge_input, edge_indices = [v.numpy() for v in x]

        results = [
            [np.random.random(size=(self.num_targets, )) for _ in node_input],
            [np.random.random(size=(len(n), self.importance_channels)) for n in node_input],
            [np.random.random(size=(len(e), self.importance_channels)) for e in edge_input]
        ]
        return [ragged_tensor_from_nested_numpy(v) for v in results]


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
                 # fidelity training
                 fidelity_factor: float = 0.0,
                 fidelity_funcs: t.List[t.Callable] = [],
                 # mlp tail end related arguments
                 final_units: t.List[int] = [1],
                 final_dropout_rate: float = 0.0,
                 final_activation: str = 'linear',
                 final_pooling: str = 'sum',
                 final_bias: t.Optional[list] = None,
                 regression_limits: t.Optional[t.Tuple[float, float]] = None,
                 regression_weights: t.Optional[t.Tuple[float, float]] = None,
                 regression_bins: t.Optional[t.List[t.Tuple[float, float]]] = None,
                 regression_reference: t.Optional[float] = None,
                 return_importances: bool = True,
                 use_graph_attributes: bool = False,
                 # contrastive sampling
                 contrastive_sampling_factor: float = 0.0,
                 contrastive_sampling_tau: float = 0.1,
                 positive_sampling_rate: int = 1,
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
        self.final_bias = final_bias
        self.regression_limits = regression_limits
        self.regression_weights = regression_weights
        self.regression_reference = regression_reference
        self.regression_bins = regression_bins
        self.return_importances = return_importances
        self.separate_explanation_step = separate_explanation_step
        self.use_graph_attributes = use_graph_attributes
        # Fidelity Training
        self.fidelity_factor = fidelity_factor
        self.var_fidelity_factor = tf.Variable(fidelity_factor, trainable=False)
        self.fidelity_funcs = fidelity_funcs
        # contrastive sampling
        self.contrastive_sampling_factor = contrastive_sampling_factor
        self.contrastive_sampling_tau = contrastive_sampling_tau
        self.positive_sampling_rate = positive_sampling_rate

        # ~ MAIN MESSAGE PASSING / ATTENTION LAYERS
        
        # 04.07.23 - I changed it here so that the last message passing layer has a linear layer in the hopes 
        # that this will make the intermediate graph embedding a little bit more visually interpretable.
        self.attention_activations = [activation for _ in self.units]
        # self.attention_activations[-1] = 'linear'
        
        self.attention_layers: t.List[GraphBaseLayer] = []
        for u, act in zip(self.units, self.attention_activations):
            lay = MultiHeadGATV2Layer(
                units=u,
                num_heads=self.importance_channels,
                use_edge_features=self.use_edge_features,
                activation=act,
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
        self.final_acts[-1] = 'linear'
        self.final_layers = []
        for u, act in zip(self.final_units, self.final_acts):
            lay = DenseEmbedding(
                units=u,
                activation=act,
                use_bias=True
            )
            self.final_layers.append(lay)

        self.lay_final_activation = ActivationEmbedding(self.final_activation)
        
        # 04.07.23 - We fix a bug here. Before we just created the new tf.variables without a condition but 
        # this has lead to problems when we load a model from the disk where the loaded bias would not be loaded 
        # properly. Now we pass in a !=None bias value ONLY when loading a model from the disk which.
        if final_bias is None:
            # As it is custom with bias variables we initialize the bias as zeros.
            self.bias = tf.Variable(tf.zeros(shape=(self.final_units[-1], )), dtype=tf.float32, name='final_bias')
        else:
            self.bias = tf.Variable(tf.constant(final_bias), dtype=tf.float32, name='final_bias')


        # ~ EXPLANATION ONLY TRAIN STEP
        self.bce_loss = ks.losses.BinaryCrossentropy()
        self.compiled_classification_loss = compile_utils.LossesContainer(bce)

        self.mse_loss = ks.losses.MeanSquaredError()
        self.mae_loss = ks.losses.MeanAbsoluteError()
        self.compiled_regression_loss = compile_utils.LossesContainer(mae)
        
        self.epsilon = 1e-8
        self.lay_contrast_dropout = DropoutEmbedding(rate=0.01)

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
            
            if final_bias is None:
                self.bias = tf.Variable(tf.constant(self.regression_reference), dtype=tf.float32, name='final_bias')

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
            "final_bias": self.final_bias,
            "regression_limits": self.regression_limits,
            "regression_weights": self.regression_weights,
            "regression_reference": self.regression_reference,
            "return_importances": self.return_importances,
            "use_graph_attributes": self.use_graph_attributes,
            "fidelity_factor": self.fidelity_factor,
            "fidelity_funcs": self.fidelity_funcs,
        })

        return config

    # ~ Properties

    @property
    def doing_regression(self) -> bool:
        return (self.regression_limits is not None) or (self.regression_weights is not None)

    def doing_regression_weights(self) -> bool:
        return self.regression_weights is not None

    @property
    def graph_embedding_shape(self) -> t.Tuple[int, int]:
        """
        Returns a tuple which defines contains the information about the shape of the graph embeddings (K, D)
        where K is the number of explanation channels employed in the model and D is the number of elements in 
        each of the embedding vectors for each of the explanation channels.
        
        Note that every explanation channel produces it's own graph embeddings!
        
        :returns: int
        """
        return self.importance_channels, self.units[-1]

    # ~ Forward Pass Implementation

    def call(self,
             inputs,
             training: bool = False,
             return_importances: bool = False,
             return_embeddings: bool = False,
             node_importances_mask: t.Optional[tf.RaggedTensor] = None,
             stop_mlp_gradient: bool = False,
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
            self.lay_sparsity(node_importances_tilde)
            self.lay_sparsity(edge_importances)
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

        if stop_mlp_gradient:
            x = tf.stop_gradient(x)
            node_importances = tf.stop_gradient(node_importances)

        embeddings = []

        # Here we apply the global pooling. It is important to note that we do K separate pooling operations
        # were each time we use the same node embeddings x but a different slice of the node importances as
        # the weights! We concatenate all the individual results in the end.
        outs = []
        out_sum = 0
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
            embeddings.append(tf.expand_dims(out, axis=-2))

            # Now "out" is a graph embedding vector of known dimension so we can simply apply the normal dense
            # mlp to get the final output value.
            num_final_layers = len(self.final_layers)
            for c, lay in enumerate(self.final_layers):
                out = lay(out)
                if training and c < num_final_layers - 2:
                    out = self.lay_final_dropout(out, training=training)

            out_sum += out
            outs.append(out)
            
        # 17.06.23 - This is a ragged tensor which contains the full graph embeddings for all the elements, these 
        # graph embeddings are essentially constant size vectors which represent the graph in some manner.
        # Note that every explanation channel k in (0..K) produces it's own graph embedding, which is reflected 
        # in the following shape information. 
        # embeddings: ([B], K, D)
        embeddings = tf.concat(embeddings, axis=-2)

        # At this point, after the global pooling of the node embeddings, we can append the global graph
        # attributes, should those exist
        # if self.use_graph_attributes:
        #     out = self.lay_concat_out([out, graph_input])

        out = self.lay_final_activation(out_sum + self.bias)

        if return_embeddings:
            return out, node_importances, edge_importances, embeddings
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

    def train_step_fidelity(self, x, out_pred, ni_pred, ei_pred):
        """
        This is an additional training step function. It will calculate a loss, it's gradients and apply
        the weight updates on the network.

        The loss that is implemented here is the "fidelity loss". The model will first perform a forward
        pass with normally with the input ``x`` and then it will perform additional forward passes for
        each of the importance channels of the model, where in each step a leave-one-in mask will be
        applied to that corresponding importance channel during the inference. The loss will then be
        calculated based on the difference of the prediction that is caused by the masking operation.

        Generally, one wants to maximize these masking differences into different directions which
        is basically the same as a large fidelity of that channel.

        :returns: The loss value
        """
        ones = tf.reduce_mean(tf.ones_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)
        zeros = tf.reduce_mean(tf.zeros_like(ni_pred, dtype=tf.float32), axis=-1, keepdims=True)

        with tf.GradientTape() as tape:
            loss = 0
            out_pred, ni_pred, ei_pred = self(x, training=True, return_importances=True)
            # ! The specific function to calculate the difference for each channel and then also compute
            # the loss from can be custom defined by the user in ``fidelity_funcs``.
            for channel_index, func in enumerate(self.fidelity_funcs):
                mask = [ones if i == channel_index else zeros for i in range(self.importance_channels)]
                mask = tf.concat(mask, axis=-1)
                out_mod, _, _ = self(
                    x,
                    training=True,
                    return_importances=True,
                    node_importances_mask=mask,
                )
                diff = func(out_pred, out_mod)
                loss += tf.reduce_mean(diff)

            loss *= self.var_fidelity_factor

        # So what we do here is we only want to train the weights which are not part of the final MLP tail
        # end! We only want to train the weights of the convolutional and importance layers with this
        # loss. Because if we could use the MLP tail end as well then the network could completely "cheat".
        mlp_vars = [weight.name for lay in self.final_layers for weight in lay.weights]
        mlp_vars = []
        trainable_vars = [var for var in self.trainable_variables if var.name not in mlp_vars]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss

    def train_step(self, data):
        """
        This method will be called to perform each train step during the model training process, which is 
        executed once for every training batch. This method will do model forward passes within an 
        automatic differentiation environment, generate the loss gradients and perform model weight update.
        
        :returns: the metrics
        """
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        exp_metrics = {'exp_loss': 0}
        with tf.GradientTape() as tape:
            exp_loss = 0
            fid_loss = 0

            node_input, edge_input, edge_indices = x[:3]
            out_true, ni_true, ei_true = y
            # 04.07.23 - I am now returning the graph embeddings here as well because I want to try to implement 
            # the unsupervised contrastive loss.
            out_pred, ni_pred, ei_pred, graph_embeddings = self(x, training=True, return_importances=True, return_embeddings=True)
            loss = self.compiled_loss(
                [out_true, ni_true, ei_true],
                [out_pred, ni_pred, ei_pred],
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            if self.importance_factor != 0 and not self.separate_explanation_step:
                # ~ APPROX. EXPLANATION LOSS
                # The idea here is that 
                
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
                    _out_pred = shifted_sigmoid(
                        outs,
                        shift=self.importance_multiplier,
                        multiplier=1
                    ) * out_true
                    exp_loss = self.compiled_classification_loss(out_true, _out_pred)

                loss += self.importance_factor * exp_loss
               
            # ~ CONTRASTIVE LOSS REGULARIZATION
            # As an additional regularization, to improve the properties of the graph embedding space, I want to try 
            # contrastive sampling/loss. This is a (mostly) unsupervised method that can be used to improve the latent 
            # data representations.
            # The core idea is that - through data augementation - we push each embedding dissimilar elements (negative) 
            # sampling and pull them closer towards similar ones.
            # https://lilianweng.github.io/posts/2021-05-31-contrastive/#common-setup
            if self.contrastive_sampling_factor != 0:
                
                batch_size = tf.shape(graph_embeddings)[0]
                virtual_batch_size = batch_size * self.importance_channels
                # This reshaping operation makes it such that we treat the separate graph embeddings of each of the 
                # channels as their own embeddings. Essentially merging the first (batch) and the second (channel) 
                # dimension to multiply the effective samples in our batch.
                # graph_embeddings: ([B * K], D)
                graph_embeddings = tf.reshape(graph_embeddings, shape=(-1, self.graph_embedding_shape[-1]))
                
                # ~ L2 normalization
                # Here we apply an L2 normalization on the embeddings, this will gradually cause the actual embedding 
                # values to become numerically very small values. This is absolutely necessary from a technical perspective 
                # because later on we are using an exponential operation essentially on the embedding norms and if these 
                # norms are too big, the exponential operation will result in infinity and break the training process.
                graph_embeddings = tf.math.l2_normalize(graph_embeddings, axis=-1)
                
                # ~ negative sampling
                # For the negative sampling we simply need to create the multiplication of every graph embedding with 
                # every other graph embedding. The following code leverages "broadcasting" to achieve just that.
                graph_embeddings_expanded = tf.expand_dims(graph_embeddings, axis=-2)
                # graph_embeddings_product: ([B], [B])
                graph_embeddings_sim = tf.reduce_sum(graph_embeddings * graph_embeddings_expanded, axis=-1)  
                #graph_embeddings_sim = 2 - tf.sqrt(tf.reduce_sum(tf.square(graph_embeddings - graph_embeddings_expanded), axis=-1) + self.epsilon)

                out_repeated = tf.repeat(out_true, axis=0, repeats=self.importance_channels)
                out_normalized = out_repeated / tf.reduce_max(out_repeated)
                out_expanded = tf.expand_dims(out_normalized, axis=-2)
                loss_debias = tf.square(1 - tf.reduce_sum(tf.abs(out_normalized - out_expanded), axis=-1))

                # Now the problem is that we do not want to consider the terms where the elements are being multiplied 
                # with themselves. So we mask them out by creating a diagonal mask of zeros here.
                eye = tf.cast(1.0 - tf.eye(num_rows=virtual_batch_size, num_columns=virtual_batch_size), tf.float32)
                #graph_embeddings_product = graph_embeddings_product * eye
                graph_embeddings_sim = graph_embeddings_sim * eye
                
                # The rest is the formula given in the paper
                exponentials = tf.reduce_sum(tf.exp(graph_embeddings_sim / self.contrastive_sampling_tau), axis=-1)
                loss_contribs = tf.math.log(exponentials)
                loss_neg = tf.reduce_mean(loss_contribs)
                
                # In the first few iterations before the l2 normalization kicks in, this loss is basically guaranteed to 
                # be infinity. This is why we have to block the loss from influencing the weight updates in those cases.
                loss += self.contrastive_sampling_factor * loss_neg
                
                # ~ positive sampling
                # The idea is that we also need positive samples as anchor points. These positive samples are supposed to 
                # be examples of graph embeddings that are CLOSE to the original ones in the batch. We can obtain these 
                # by data augmentation techniques. For example we can make the assumption that despite small perturbations 
                # of the input graphs, the graph embedding should still be functionally the same.
                for p in range(self.positive_sampling_rate):
                    # first of all we create the perturbations for the input graphs and then we make another forward pass 
                    # with those inputs.
                    
                    # We can use this somewhat complicated code here to produce a random binary mask with the same shape as 
                    # the predicted node importances tensor. We effectively use this binary mask to drop out only a very few
                    # importances annotations.
                    node_importances_mask = tf.map_fn(
                        lambda tens: tf.cast(
                            tf.random.categorical(
                                tf.math.log(
                                    tf.repeat(
                                        [[0.1, 0.9]], 
                                        repeats=tf.shape(tens)[0],
                                        axis=0)
                                    ), 
                                    num_samples=self.importance_channels
                                ), 
                            tf.float32
                        ),
                        ni_pred,
                        dtype=tf.RaggedTensorSpec(shape=(None, self.importance_channels), dtype=tf.float32, ragged_rank=0)
                    )
                    _, _, _, graph_embeddings_pos =  self(
                        x,
                        training=True, 
                        return_importances=True, 
                        return_embeddings=True,
                        node_importances_mask=node_importances_mask,
                    )
                    
                    # The resulting graph embeddings here we have to process in the same manner as we have processed the 
                    # the original graph embeddings as well, which means flattening the channel dimension and l2 normalize
                    graph_embeddings_pos = tf.reshape(graph_embeddings_pos, shape=(-1, self.graph_embedding_shape[-1]))
                    graph_embeddings_pos = tf.math.l2_normalize(graph_embeddings_pos, axis=-1)
                    
                    loss_contribs_pos = tf.reduce_sum(graph_embeddings * graph_embeddings_pos, axis=-1)
                    #loss_contribs_pos = 2 - tf.sqrt(tf.reduce_sum(tf.square(graph_embeddings - graph_embeddings_pos), axis=-1) + self.epsilon)
                    loss_pos = tf.reduce_mean(loss_contribs_pos / self.contrastive_sampling_tau)
                    loss -= (self.contrastive_sampling_factor / self.positive_sampling_rate) * loss_pos
                    
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # ~ FIDELITY TRAIN STEP
        # 09.05.23 - Optionally, we execute another additional train step here which can be used to
        # directly train the fidelity contributions of each of the channels to behave according to some
        # given functions. Specifically the thing that is being trained here is the difference between
        # the original prediction another new prediction in a leave-one-in channel masking.
        if self.fidelity_factor != 0:
            fidelity_loss = self.train_step_fidelity(x, out_pred, ni_pred, ei_pred)
            exp_loss += fidelity_loss

        exp_metrics['exp_loss'] = exp_loss
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

    def embedd_graphs(self,
                      graph_list: t.List[tv.GraphDict],
                      concat_channels: bool = False,
                      **kwargs
                      ) -> np.ndarray:
        """
        Given a ``graph_list`` list of graph dicts, this method will return a numpy array which contains the 
        graph embeddings. The first dimension of the numpy array will contain the graph embeddings in the same 
        order as the order of the graphs in the list.
        
        :param graph_list: The list of graph dicts.
        :param concat_channels: boolean flag. If True, all the embeddings of the individual explanation channels 
            will be concatenated into a single embedding.
        
        :returns: Array of shape ([B], K, N) if concat_channels is False and ([B], K*N) if it is True.
        """
        x = tensors_from_graphs(graph_list)
        
        # Only when setting the "return_embeddings" flag do we change the output signature of the call method 
        # so that it now returns 4 elements where the last one is the additional graph embeeding vector with 
        # the following shape.
        # embeddings: ([B], K, D)
        _, _, _, embeddings = self(x, return_embeddings=True)
        
        return embeddings.numpy()

    def predict_graphs(self,
                       graph_list: t.List[tv.GraphDict],
                       **kwargs
                       ) -> t.Any:
        """
        Given a list of GraphDicts, returns the predictions of the network. The output will be a list
        consisting of a tuple for each of the input graphs. Each of these tuples consists of 3 values:
        (prediction, node_importances, edge_importances)

        :returns: list
        """
        x = tensors_from_graphs(graph_list)
        return list(zip(*[v.numpy() for v in self(x)]))

    def predict_graph(self, graph: tv.GraphDict):
        """
        Predicts the output for a single GraphDict.
        """
        return self.predict_graphs([graph])[0]

    # -- Implements "FidelityGraphMixin"

    def leave_one_out_deviations(self,
                                 graph_list: t.List[tv.GraphDict],
                                 ) -> np.ndarray:
        """
        Given a list of graphs, this method will compute the explanation leave-one-out deviations.
        This is done by making an initial prediction for the given graphs and then for each explanation
        channel which the model employs an additional prediction where that corresponding explanation
        channel is masked such that all of it's information is withheld from the final prediction result.
        The method will return a numpy array of the shape (N, K, C) where N is the number of graphs given
        to the method, K is the number of importance channels of the model and C is the number of output
        values generated by the model. Each element in this array will be the deviation (original - modified)
        that is caused for the c-th output value when the k-th importance channel is withheld for the
        n-th graph.

        :param graph_list: A list of GraphDicts for which to compute this.

        :returns: Array of shape (N, K, C)
        """
        x = tensors_from_graphs(graph_list)
        y_org = self(x, training=False)
        out_org, _, _ = [v.numpy() for v in y_org]

        num_channels = self.importance_channels
        num_targets = self.final_units[-1]

        results = np.zeros(shape=(len(graph_list), num_channels, num_targets), dtype=float)
        for channel_index in range(self.importance_channels):
            base_mask = [float(channel_index != i) for i in range(self.importance_channels)]
            mask = [[base_mask for _ in graph['node_indices']] for graph in graph_list]
            mask_tensor = ragged_tensor_from_nested_numpy(mask)
            y_mod = self(x, training=False, node_importances_mask=mask_tensor)
            out_mod, _, _ = [v.numpy() for v in y_mod]

            for target_index in range(num_targets):
                for index, out in enumerate(out_mod):
                    deviation = out_org[index][target_index] - out_mod[index][target_index]
                    results[index, channel_index, target_index] = deviation

        return results

    def leave_one_out(self,
                      graph_list: t.List[tv.GraphDict],
                      channel_funcs: t.Optional[t.List[t.Callable]] = None,
                      **kwargs) -> np.ndarray:
        """
        Given a list of GraphDict's as input elements to the network, this method will calculate the
        fidelity value for each of those input elements and for each of the importance channels of the
        network, returning a numpy array of the shape (num_elements, num_channels).

        :param graph_list: A list of GraphDicts to be used as inputs for the fidelity calculation.
        :param channel_funcs: This needs to be a list with as many elements as there are importance channels
            used in this model. Each element of the list should be a function that defines how the
            fidelity for that channel is calculated. Each function gets as the input the original predicition
            and the modified prediction and is supposed to return a single float value that represents
            that channels fidelity contribution.

        :returns: numpy array
        """
        x = tensors_from_graphs(graph_list)
        y_org = self(x, training=False)
        out_org, _, _ = [v.numpy() for v in y_org]

        results = np.zeros(shape=(len(graph_list), self.importance_channels), dtype=float)
        for channel_index in range(self.importance_channels):
            base_mask = [float(channel_index != i) for i in range(self.importance_channels)]
            mask = [[base_mask for _ in graph['node_indices']] for graph in graph_list]
            mask_tensor = ragged_tensor_from_nested_numpy(mask)
            y_mod = self(x, training=False, node_importances_mask=mask_tensor)
            out_mod, _, _ = [v.numpy() for v in y_mod]

            for index, out in enumerate(out_mod):
                if channel_funcs is None:
                    fidelity = out_org[index] - out_mod[index]
                else:
                    fidelity = channel_funcs[channel_index](out_org[index], out_mod[index])

                results[index, channel_index] = fidelity

        return results

