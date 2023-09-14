import os 

import tensorflow as tf
import tensorflow.keras as ks


def tf_cosine_sim(tens1: tf.Tensor, tens2: tf.Tensor, normalize: bool = False):
    
    if normalize:
        tens1 = tf.math.l2_normalize(tens1, axis=-1)
        tens2 = tf.math.l2_normalize(tens2, axis=-1)
        
    sim = tf.reduce_sum(tens1 * tens2, axis=-1)
    return sim


def tf_pairwise_cosine_sim(tens1: tf.Tensor, tens2: tf.Tensor, normalize: bool = False):
    
    if normalize:
        tens1 = tf.math.l2_normalize(tens1, axis=-1)
        tens2 = tf.math.l2_normalize(tens2, axis=-1)
    
    # tens1: (A, B)
    # tens2: (A, B)
    # tens_expanded: (A, 1, B)
    tens_expanded = tf.expand_dims(tens2, axis=-2)
    
    # sim: (A, A)
    sim = tf.reduce_sum(tens1 * tens_expanded, axis=-1)
    
    return sim


def tf_pairwise_isc_sim(tens1: tf.Tensor, tens2: tf.Tensor, normalize: bool = False):
    
    tens1_norm = tf.linalg.normalize(tens1, ord=1, axis=-1)
    tens2_norm = tf.linalg.normalize(tens2, ord=1, axis=-1)
    
    # tens1: (A, B)
    # tens2: (A, B)
    # tens_expanded: (A, 1, B)
    tens2_expanded = tf.expand_dims(tens2, axis=-2)
    
    # sim: (A, A)
    sim = tf.reduce_sum(tf.sqrt((tens1 / tens1_norm) * (tens2_expanded / tens2_norm)), axis=-1)
    
    return sim


def tf_cauchy_sim(tens1: tf.Tensor, tens2: tf.Tensor):
    dist = tf.reduce_sum(tf.square(tens1 - tens2), axis=-1)
    sim = 1.0 / (1.0 + dist)
    return sim


def tf_pairwise_cauchy_sim(tens1: tf.Tensor, tens2: tf.Tensor):
    # tens1: (A, B)
    # tens2: (A, B)
    # tens_expanded: (A, 1, B)
    tens_expanded = tf.expand_dims(tens2, axis=-2)
    
    # dist: (A, A)
    dist = tf.reduce_sum(tf.math.pow(tens1 - tens_expanded, 2), axis=-1)
    
    # sim: (A, A)
    sim = 1.0 / (1.0 + dist)
    
    return sim


def tf_euclidean_distance(tens1: tf.Tensor, tens2: tf.Tensor):
    # tens1: (A, B)
    # tens2: (A, B)
    dist = tf.reduce_sum(tf.square(tens1 - tens2), axis=-1)

    return dist


def tf_pairwise_euclidean_distance(tens1: tf.Tensor, tens2: tf.Tensor):
    # tens1: (A, B)
    # tens2: (A, B)
    # tens_expanded: (A, 1, B)
    tens_expanded = tf.expand_dims(tens2, axis=-2)
    
    # sim: (A, A)
    dist = tf.sqrt(tf.reduce_sum(tf.square(tens1 - tens_expanded), axis=-1))
    
    return dist


def tf_manhattan_distance(tens1: tf.Tensor, tens2: tf.Tensor):
    # tens1: (A, B)
    # tens2: (A, B)
    dist = tf.reduce_sum(tf.abs(tens1 - tens2), axis=-1)

    return dist


def tf_pairwise_manhattan_distance(tens1: tf.Tensor, tens2: tf.Tensor):
    # tens1: (A, B)
    # tens2: (A, B)
    # tens_expanded: (A, 1, B)
    tens_expanded = tf.expand_dims(tens2, axis=-2)
    
    # sim: (A, A)
    dist = tf.sqrt(tf.reduce_sum(tf.abs(tens1 - tens_expanded), axis=-1))
    
    return dist


def tf_pairwise_variance(tens1: tf.Tensor, tens2: tf.Tensor):
    # tens1: (A, B)
    # tens2: (A, B)
    # tens_expanded: (A, 1, B)
    tens_expanded = tf.expand_dims(tens2, axis=-2)
    
    # sim: (A, A)
    dist = tf.math.reduce_variance(tens1 - tens_expanded, axis=-1)
    
    return dist


def tf_ragged_random_binary_mask(template: tf.RaggedTensor,
                                 chances: list = [0.5, 0.5],
                                 n: int = 1,
                                 ) -> tf.RaggedTensor:
    # template: ([A], [B], C)
    # mask: ([A], [B], N)
    mask = tf.map_fn(
        lambda tens: tf.cast(
            tf.random.categorical(
                tf.math.log(
                    tf.repeat(
                        [chances], 
                        repeats=tf.shape(tens)[0],
                        axis=0)
                    ), 
                    num_samples=n,
                ), 
            tf.float64
        ),
        template,
        dtype=tf.RaggedTensorSpec(shape=(None, n), dtype=tf.float64, ragged_rank=0)
    )
    
    return mask


class EpochCounterCallback(ks.callbacks.Callback):

    def __init__(self):
        super(EpochCounterCallback, self).__init__()
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1
        
        
class ChangeVariableCallback(EpochCounterCallback):
    
    def __init__(self,
                 property_name: str,
                 start_value: float,
                 end_value: float,
                 epoch_threshold: int
                 ):
        super(ChangeVariableCallback, self).__init__()
        
        self.property_name = property_name 
        self.start_value = start_value
        self.end_value = end_value
        self.epoch_threshold = epoch_threshold

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch >= self.epoch_threshold:
            self.set_value(self.end_value)
        else:
            self.set_value(self.start_value)
    
    def set_value(self, value: float):
        var = getattr(self.model, self.property_name)
        var.assign(value)