import logging
import time
import typing as t

import tensorflow as tf
import tensorflow.keras as ks

from graph_attention_student.util import NULL_LOGGER


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