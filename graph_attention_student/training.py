import time
import types
import logging
import typing as t
from typing import List, Dict, Optional, Callable
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as ks


# == CUSTOM ACTIVATION FUNCTIONS ============================================================================
# ===========================================================================================================

def shifted_sigmoid(x, multiplier: float = 1, shift: float = 10) -> float:
    return ks.backend.sigmoid(multiplier * (x - shift))


# == TRAINING SEGMENTATION ==================================================================================
# ===========================================================================================================


class RecompilableMixin:

    def __init__(self):
        self.compile_args: Optional[list] = None
        self.compile_kwargs: Optional[dict] = None

    def compile(self, *args, **kwargs):
        ks.models.Model.compile(self, *args, **kwargs)
        self.compile_args = args
        self.compile_kwargs = kwargs

    def recompile(self):
        self.compile(*self.compile_args, **self.compile_kwargs)


class SegmentedFitProcess:

    def __init__(self,
                 model: ks.models.Model,
                 fit_kwargs: dict):
        self.model = model
        self.fit_kwargs = fit_kwargs

        self.total_epochs = fit_kwargs['epochs']
        self.callbacks = dict()
        self.callbacks[self.total_epochs] = lambda model, history: None
        self.current_epoch = 0

    def __call__(self):
        """
        This method actually executes the various model.fit() operations which are part of the overall
        segmented process. It then returns a ks History object which contains the *overall* history which
        consists of the merged histories of all the individual processes.

        :return:
        """
        hists = []

        # self.callbacks: Dict[int, Callable] is a dictionary whose keys are integer epoch numbers which
        # are generally between 0 and the total number of epochs defined for the entire fit process.
        # The corresponding values are callbacks which are applied *on the model* at the point at which the
        # overall process hits that epoch.
        for epoch, cb in sorted(self.callbacks.items(), key=lambda i: i[0]):
            # The next fit process has to have as many epochs as starting from the current one and until the
            # next callback has to be applied.
            epoch_diff = epoch - self.current_epoch

            # For each individual fit process we can just reuse the majority of the fit kwargs, but we need
            # to replace the epoch count with the local count only until the next callback is due.
            fit_kwargs = self.fit_kwargs.copy()
            fit_kwargs['epochs'] = epoch_diff

            # We train the model and then after it is done we apply the callback.
            hist = self.model.fit(**fit_kwargs)
            hists.append(hist)
            cb(
                model=self.model,
                history=hist
            )

            self.current_epoch += epoch_diff

        merged_hist = self.merge_histories(hists)
        return merged_hist

    def merge_histories(self, hists: List[ks.callbacks.History]):
        merged_hist = ks.callbacks.History()

        merged_history = defaultdict(list)
        for hist in hists:
            for key, value in hist.history.items():
                if isinstance(value, list):
                    merged_history[key] += value

        merged_hist.history = merged_history
        return merged_hist

    def add_callback(self,
                     epochs: int,
                     callback: Callable[[ks.models.Model, ks.callbacks.History], None]
                     ) -> None:
        self.callbacks[epochs] = callback

# This section is a lot of black magic
# ------------------------------------

def epochs_once(epoch: int):

    def register(self, instance, fit_process, fit_kwargs):
        # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
        bound_method = self.__get__(instance, instance.__class__)
        fit_process.add_callback(epoch, bound_method)

    def local(func: Callable):
        method = types.MethodType(register, func)
        setattr(func, 'register', method)
        setattr(func, 'is_epoch_callback', True)

        return func

    return local


class FitManager:

    def __init__(self):
        pass

    def __call__(self,
                 model: ks.models.Model,
                 fit_kwargs: dict) -> SegmentedFitProcess:
        fit_process = SegmentedFitProcess(model, fit_kwargs)
        # self.register_callbacks(fit_process, fit_kwargs, model)
        callbacks = self.gather_callbacks()
        for callback in callbacks:
            callback.register(self, fit_process, fit_kwargs)

        return fit_process

    def gather_callbacks(self) -> List[Callable]:
        callbacks = []
        for name in dir(self):
            attribute = getattr(self, name)
            if callable(attribute) and hasattr(attribute, 'is_epoch_callback'):
                callbacks.append(attribute)

        return callbacks

    def register_callbacks(self,
                           fit_process: SegmentedFitProcess,
                           fit_kwargs: dict,
                           model: ks.models.Model) -> None:
        return


class LockExplanationManager(FitManager):

    def __init__(self,
                 epoch_threshold: int):
        FitManager.__init__(self)
        self.epoch_threshold = epoch_threshold

    def register_callbacks(self, fit_process, fit_kwargs, model):
        fit_process.add_callback(self.epoch_threshold, self.lock_model_explanation_parameters)

    def lock_model_explanation_parameters(self, model, history):
        model.lock_explanation_parameters()
        model.recompile()


class NoLoss(ks.losses.Loss):

    def __init__(self,
                 *args,
                 name='no_loss',
                 **kwargs):
        super(NoLoss, self).__init__(*args, name=name, **kwargs)

    def call(self, y_true, y_pred):
        return 0


def bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = y_true * tf.math.log(y_pred + 1e-7)
    loss += (1 - y_true) * tf.math.log(1 - y_pred + 1e-7)
    loss = -loss
    return tf.reduce_mean(loss)


def mae(y_true: t.Union[tf.Tensor, tf.RaggedTensor],
        y_pred: t.Union[tf.Tensor, tf.RaggedTensor],
        ) -> float:
    """
    A custom implementation of the "mean absolute error" loss.

    Unlike the default keras implementation, this custom implementation supports two additional features:
    1. It is able to handle RaggedTensors properly
    2. It is able to handle NaN values in y_true. This is especially important for multitask learning where
       not every element may have a defined target value for all targets. NaN values will be handled such
       that there will result NO gradients from those elements.

    :param y_true: The tensor of ground truth values
    :param y_pred: The tensor of predicted values from the model

    :returns: A single float loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    # 01.04.2023 - Changed the way in which the nan_multiplier is created here because it turned out that
    # the previous method with tf.where was not supported for RaggedTensors.
    # nan_multiplier = tf.where(tf.math.is_nan(y_true), 0., 1.)
    nan_multiplier = tf.cast(tf.math.is_nan(y_true), tf.float32)
    nan_multiplier = tf.ones_like(nan_multiplier) - nan_multiplier

    loss = tf.abs(y_true - y_pred)

    loss *= nan_multiplier
    loss = tf.reduce_sum(loss, axis=-1)

    return tf.reduce_mean(loss)


mae.name = 'mean_absolute_error'


def mse(y_true: t.Union[tf.Tensor, tf.RaggedTensor],
        y_pred: t.Union[tf.Tensor, tf.RaggedTensor],
        ) -> float:
    """
    A custom implementation of the "mean squared error" loss.

    Unlike the default keras implementation, this custom implementation supports two additional features:
    1. It is able to handle RaggedTensors properly
    2. It is able to handle NaN values in y_true. This is especially important for multitask learning where
       not every element may have a defined target value for all targets. NaN values will be handled such
       that there will result NO gradients from those elements.

    :param y_true: The tensor of ground truth values
    :param y_pred: The tensor of predicted values from the model

    :returns: A single float loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    # 01.04.2023 - Changed the way in which the nan_multiplier is created here because it turned out that
    # the previous method with tf.where was not supported for RaggedTensors.
    # nan_multiplier = tf.where(tf.math.is_nan(y_true), 0., 1.)
    nan_multiplier = tf.cast(tf.math.is_nan(y_true), tf.float32)
    nan_multiplier = tf.ones_like(nan_multiplier) - nan_multiplier
    y_true = tf.math.multiply_no_nan(y_true, nan_multiplier)

    loss = tf.square(y_true - y_pred)

    loss *= nan_multiplier
    loss = tf.reduce_sum(loss, axis=-1)
    loss_reduced = tf.reduce_mean(loss)

    return loss_reduced

mse.name = 'mean_squared_error'


class ExplanationLoss(ks.losses.Loss):
    """
    Implementation of keras Loss specifically for attributional explanations.

    The thing which makes the loss for attributional explanations special is that the corresponding
    ground truth and predicted tensors are tf.RaggedTensors, which are made up of graphs with different
    sizes.

    :param loss_function: A python function which accepts two positional parameters (y_true, y_pred) and
        returns a single float loss value. NOTE: This function must explicitly support RaggedTensors, which
        is usually not the case for the keras built-in loss functions!
    :param mask_empty_explanations: A boolean flag of whether completely empty explanations should be masked
        out completely. If this is True, a completely empty ground truth explanation will not be used to
        train the predicted explanations to be empty as well, but rather all gradients will be stopped
        for that case entirely, such that it does not affect the model training process at all.
    :param reduce: A boolean flag of whether the last explanation dimension should be reduced before
        attempting to calculate the loss. Default is False.
    :param factor: An additional float value which is multiplied with the final loss value. Default is 1.0
    """
    def __init__(self,
                 loss_function: ks.losses.Loss = bce,
                 mask_empty_explanations: bool = False,
                 reduce: bool = False,
                 factor: float = 1):
        ks.losses.Loss.__init__(self)

        self.loss_function = loss_function
        self.mask_empty_explanations = mask_empty_explanations
        self.factor = factor
        self.reduce = reduce
        self.name = 'explanation_loss'

    def call(self, y_true, y_pred):
        loss = self.loss_function(y_true, y_pred)

        if self.mask_empty_explanations:
            mask = tf.cast(tf.reduce_max(y_true, axis=-1) > 0, dtype=tf.float32)
            loss *= mask

        if self.reduce:
            loss = tf.reduce_mean(loss, axis=-1)

        return self.factor * loss


class EpochCounterCallback(ks.callbacks.Callback):

    def __init__(self):
        super(EpochCounterCallback, self).__init__()
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1


class LogProgressCallback(EpochCounterCallback):

    def __init__(self,
                 logger: logging.Logger,
                 identifier: str,
                 epoch_step: int):
        super(LogProgressCallback, self).__init__()

        self.logger = logger
        self.identifier = identifier
        self.epoch_step = epoch_step

        self.start_time = time.time()
        self.elapsed_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch % self.epoch_step == 0 or self.epoch == 1) and logs is not None:
            self.elapsed_time = time.time() - self.start_time
            print(logs)
            value = logs[self.identifier]
            

            message_parts = [
                f'   epoch {str(self.epoch):<5}: ',
                f'{self.identifier}={value:.3f} ',
                f'elapsed_time={self.elapsed_time:.1f}s'
            ]
            if 'output_1_loss' in logs:
                message_parts.append(f'output_1_loss={logs["output_1_loss"]:.3f} ')
            if 'output_2_loss' in logs:
                message_parts.append(f'output_2_loss={logs["output_2_loss"]:.3f}')
            if 'output_3_loss' in logs:
                message_parts.append(f'output_3_loss={logs["output_3_loss"]:.3f}')
            if 'loss' in logs:
                message_parts.append(f'total_training_loss={logs["loss"]:.3f} ')
            if 'exp_loss' in logs:
                message_parts.append(f'exp_loss={logs["exp_loss"]:.3f}')

            self.logger.info(' '.join(message_parts))

    def reset(self):
        self.epoch = 0
        self.start_time = time.time()
        self.elapsed_time = 0