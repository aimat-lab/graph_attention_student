import logging
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def asym_binary_cross_entropy(inputs: torch.Tensor,
                              targets: torch.Tensor,
                              weight_0: float = 1.5,
                              weight_1: float = 1.0,
                              reduction: str = 'mean'
                              ) -> torch.Tensor:
    """
    Calculates asymmetric binary cross entropy loss with different weights for positive and negative samples.
    By default assigns 50% more weight to the "0" class (negative samples).

    :param inputs: Predicted logits/probabilities
    :param targets: Ground truth binary labels (0 or 1)
    :param weight_0: Weight for class 0 (negative samples), default 1.5
    :param weight_1: Weight for class 1 (positive samples), default 1.0
    :param reduction: Reduction method ('mean', 'sum', or 'none')
    :returns: Computed asymmetric BCE loss
    """
    # Convert logits to probabilities if needed
    probs = torch.sigmoid(inputs)

    # Calculate BCE for each sample
    bce_0 = -targets * torch.log(probs + 1e-8)  # Loss for class 1
    bce_1 = -(1 - targets) * torch.log(1 - probs + 1e-8)  # Loss for class 0

    # Apply asymmetric weights
    weighted_loss = weight_1 * bce_0 + weight_0 * bce_1

    if reduction == 'mean':
        return torch.mean(weighted_loss)
    elif reduction == 'sum':
        return torch.sum(weighted_loss)
    else:
        return weighted_loss


def binary_entropy_reg(x, p=0.5):
    # H(x) = -x*log(x) - (1-x)*log(1-x)
    # Maximum at x=0.5, minimum at x=0 or x=1
    
    return (x.abs() + 1e-8).pow(p)

def hoyer_square_reg(x, epsilon=1e-8):
    
    # Flatten the tensor
    x_flat = x.view(-1)
    
    # Mean of absolute values
    l1_mean = torch.mean(torch.abs(x_flat))
    
    # Mean of squares
    l2_mean_squared = torch.mean(x_flat ** 2) + epsilon
    
    # Mean-based Hoyer-Square
    hs_mean = (l1_mean ** 2) / l2_mean_squared
    
    return hs_mean


class SwaCallback(pl.Callback):
    
    def __init__(self,
                 history_length: int = 10,
                 logger: logging.Logger = NULL_LOGGER):
        super(SwaCallback, self).__init__()
        self.history_length = history_length
        self.logger = logger
        
        self.history: List[torch.Tensor] = deque(maxlen=history_length)
        
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
        
        
class FocalLoss(nn.Module):
    
    def __init__(self, 
                 alpha=0.9,  # between 0 and 1, balancing parameter
                 gamma=2, # between 1 and 5, focusing/difficulty parameter
                 reduction='mean'
                 ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
        
        
class EvidentialLoss(nn.Module):
    """
    EvidentialLoss computes a loss function based on evidential deep learning.

    This class calculates a loss that not only measures the prediction error but also the
    uncertainty by leveraging the evidence provided by the network outputs. It combines a
    squared error term with a regularization term derived from the KL divergence between the
    predicted Dirichlet distribution and a uniform prior.

    Background:
        Evidential deep learning provides a framework where besides predicting class probabilities,
        a model quantifies uncertainty using evidence.

    Example:
        >>> loss_fn = EvidentialLoss(num_classes=3, reg_factor=0.01, reduction='mean')
        >>> loss = loss_fn(inputs, targets)

    :param num_classes: Number of classes in the classification task.
    :param reg_factor: Regularization factor for the evidential regularization term.
    :param reduction: Specifies the reduction applied to the output loss ('mean' or 'none').
    """
    def __init__(self, num_classes: int, reg_factor: float = 0.01, reduction: str = 'mean') -> None:
        """
        Initialize an EvidentialLoss instance.

        :param num_classes: The number of classes.
        :param reg_factor: A float specifying the regularization factor.
        :param reduction: A string specifying the loss aggregation method ('mean' or 'none').
        """
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.reg_factor = reg_factor
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the evidential loss for the given predictions and targets.

        The method first applies a ReLU to extract evidence, computes the corresponding Dirichlet
        parameters, and then calculates the loss as a combination of a squared error term and a
        regularization term based on the KL divergence.

        :param inputs: Raw outputs from the network as a tensor.
        :param targets: Ground truth labels in a one-hot encoded tensor.
        :return: Computed loss as a tensor.
        """
        # Apply ReLU to ensure evidence values are non-negative.
        evidence = F.softplus(inputs)
        # Shift evidence by 1 to obtain Dirichlet parameters.
        alpha = evidence + 1  
        # Compute the sum of the Dirichlet parameters along classes.
        S = torch.sum(alpha, dim=1, keepdim=True)
        # Normalize to get the expected probabilities.
        m = alpha / S
       
        # Compute the squared difference between targets and expected probabilities.
        A = torch.sum((targets - m) ** 2, dim=1, keepdim=True)
        # Compute the uncertainty term derived from the Dirichlet parameters.
        B = torch.sum(alpha * (1 - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

        # Combine the error and uncertainty terms with the regularization.
        loss = A + B + self.reg_factor * self.evidential_regularization(evidence)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss
        
    def kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL divergence between the predicted Dirichlet distribution and a uniform prior.

        A uniform Dirichlet prior (with ones) is constructed and the divergence is calculated 
        between it and the predicted Dirichlet distribution parameterized by alpha.

        :param alpha: Tensor of Dirichlet parameters.
        :return: Tensor representing the KL divergence.
        """
        # Create a uniform prior for Dirichlet distributions (all ones).
        beta = torch.ones([1, self.num_classes], dtype=torch.float32, device=alpha.device)
        # Compute the sum of Dirichlet parameters for the prediction and the prior.
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        # Calculate the logarithm of the beta functions using the gamma function.
        lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_beta = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
        # Compute digamma values for the Dirichlet parameters.
        dg_alpha = torch.digamma(alpha)
        dg_beta = torch.digamma(beta)
        # Compute the KL divergence based on differences in log beta functions and digamma functions.
        kl = lnB_alpha - lnB_beta + torch.sum((alpha - beta) * (dg_alpha - dg_beta), dim=1, keepdim=True)
        return kl

    def evidential_regularization(self, evidence: torch.Tensor) -> torch.Tensor:
        """
        Calculate the evidential regularization term based on the KL divergence.

        This term penalizes the deviation of the predicted Dirichlet distribution from 
        the uniform prior, helping to quantify uncertainty in the model predictions.

        :param evidence: The evidence tensor obtained from network outputs.
        :return: Mean regularization value as a tensor.
        """
        # Convert evidence to Dirichlet parameters (alpha) by shifting.
        alpha = evidence + 1
        # Compute the KL divergence between the predicted and uniform Dirichlet distributions.
        kl = self.kl_divergence(alpha)
        # Return the mean KL divergence as the regularization term.
        return torch.mean(kl)