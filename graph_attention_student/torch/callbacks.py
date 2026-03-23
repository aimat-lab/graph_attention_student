"""
PyTorch Lightning callbacks for MEGAN model training.
"""
import typing as t

from lightning.pytorch.callbacks import Callback


class ImportanceFactorWarmup(Callback):
    """
    A PyTorch Lightning callback that gradually ramps up the importance factor of a MEGAN model
    over the course of training using a linear schedule.

    The callback reads the target importance factor from the model at the start of training,
    then linearly interpolates from `start_value` to that target over `warmup_epochs` epochs.
    After the warmup period, the importance factor remains at the target value.

    :param start_value: The initial importance factor value at epoch 0. Default is 1e-6.
    :param warmup_epochs: The number of epochs over which to ramp up to the final value.

    Example usage::

        from graph_attention_student.torch.callbacks import ImportanceFactorWarmup

        model = Megan(importance_factor=1.0, ...)
        callback = ImportanceFactorWarmup(start_value=1e-6, warmup_epochs=50)
        trainer = Trainer(callbacks=[callback], max_epochs=150)
        trainer.fit(model, datamodule)
    """

    def __init__(self,
                 start_value: float = 1e-6,
                 warmup_epochs: int = 50,
                 ):
        super().__init__()
        self.start_value = start_value
        self.warmup_epochs = warmup_epochs
        self.final_value: t.Optional[float] = None

    def on_fit_start(self, trainer, pl_module) -> None:
        """
        Called at the start of training. Reads the target importance factor from the model
        and sets the initial value.
        """
        self.final_value = pl_module.importance_factor
        pl_module.importance_factor = self.start_value

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """
        Called at the start of each training epoch. Updates the importance factor
        based on the current epoch using linear interpolation.
        """
        current_epoch = trainer.current_epoch

        if current_epoch >= self.warmup_epochs:
            # Warmup complete, use final value
            pl_module.importance_factor = self.final_value
        else:
            # Linear interpolation
            progress = current_epoch / self.warmup_epochs
            pl_module.importance_factor = (
                self.start_value + progress * (self.final_value - self.start_value)
            )
