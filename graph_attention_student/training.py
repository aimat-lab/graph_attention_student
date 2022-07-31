import time
import logging

import tensorflow as tf
import tensorflow.keras as ks


class NoLoss(ks.losses.Loss):

    def __init__(self,
                 *args,
                 name='no_loss',
                 **kwargs):
        super(NoLoss, self).__init__(*args, name=name, **kwargs)

    def call(self, y_true, y_pred):
        return 0


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
        if self.epoch % self.epoch_step == 0 or self.epoch == 1:
            self.elapsed_time = time.time() - self.start_time
            value = logs[self.identifier]

            message_parts = [
                f'   epoch {str(self.epoch):<5}: ',
                f'{self.identifier}={value:.2f} ',
                f'elapsed_time={self.elapsed_time:.1f}s'
            ]
            if 'output_1_loss' in logs:
                message_parts.append(f'training_loss={logs["output_1_loss"]:.2f} ')

            self.logger.info(' '.join(message_parts))

    def reset(self):
        self.epoch = 0
        self.start_time = time.time()
        self.elapsed_time = 0