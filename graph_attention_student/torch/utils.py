import logging
from collections import deque

import torch
import pytorch_lightning as pl
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from graph_attention_student.utils import NULL_LOGGER


def torch_uniform(size: tuple,
                  lower: float,
                  upper: float
                  ) -> torch.Tensor:
    """
    Returns a torch tensor with the given shape ``size`` that contains randomly uniformly 
    distributed values in the range from ``lower`` to ``upper``.
    
    :returns: Torch Tensor
    """
    return (lower-upper) * torch.rand(size=size) + upper 


def torch_gauss(size: tuple,
                mean: float,
                std: float,
                ) -> torch.Tensor:
    """
    Returns a torch tensor with the given shape ``size`` whose values are randomly generated 
    from a normal (gauss) distribution with ``mean`` and standard deviation ``std``
    
    :returns: Torch Tensor
    """
    tens_mean = torch.full(size=size, fill_value=mean, dtype=torch.float32)
    tens_std = torch.full(size=size, fill_value=std, dtype=torch.float32)
    return torch.normal(tens_mean, tens_std)


class SwaCallback(pl.Callback):
    
    def __init__(self,
                 history_length: int = 10,
                 logger: logging.Logger = NULL_LOGGER):
        super(SwaCallback, self).__init__()
        self.history_length = history_length
        self.logger = logger
        
        self.history: list[torch.Tensor] = deque(maxlen=history_length)
        
    def on_train_start(self, trainer, pl_module):
        self.logger.info('starting to record model weights for SWA')
    
    def on_train_epoch_end(self, trainer, pl_module):
        weights = parameters_to_vector(pl_module.parameters())
        self.history.append(weights)

    def on_train_end(self, trainer, pl_module):
        self.logger.info(f'training finished, starting to compute SWA weights from {len(self.history)} recordings')
        
        weights_stacked = torch.stack(list(self.history), dim=-1)
        weights_mean = torch.mean(weights_stacked, dim=-1)
        # This utility function handles the expansion of the now flat weights vector back into the 
        # parameter dict of the actual module.
        vector_to_parameters(weights_mean, pl_module.parameters())