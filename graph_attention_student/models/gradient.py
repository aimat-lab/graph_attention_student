import typing as t

import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.layers.conv.gcn_conv import GCN
from kgcnn.layers.modules import DenseEmbedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.literature.GNNExplain import GNNInterface


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
                     use_sigmoid: bool = False,
                     keepdims: bool = False
                     ) -> tf.RaggedTensor:
    """

    """

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
    if use_sigmoid:
        importances = ks.backend.sigmoid(importances)

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
