from typing import List, Tuple
import typing as typ

import numpy as np


def sign(value: float) -> float:
    
    if value >= 0:
        return +1.0
    else:
        return -1.0
    
    
def intervals_overlap(interval_1: tuple, interval_2: tuple) -> bool:
    """
    Given the two interval tuples ``interval_1`` and ``interval_2``, this function checks if the intervals
    overlap. The intervals are defined as tuples of two values: the first value is the minimum value of the interval
    and the second value is the maximum value of the interval.
    
    :param interval_1: The first interval as a tuple of two values.
    :param interval_2: The second interval as a tuple of two values.
    
    :returns: True if the intervals overlap, False otherwise.
    """
    min_1 = min(interval_1)
    max_1 = max(interval_1)
    
    min_2 = min(interval_2)
    max_2 = max(interval_2)
    
    return max(min_1, min_2) <= min(max_1, max_2)


def threshold_error_reduction(uncertainties: np.ndarray,
                              errors: np.ndarray,
                              error_func: typ.Callable = np.mean,
                              num_bins: int = 10,
                              percentile: int = 5,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an array of ``uncertainties`` and an array of ``errors``, this function computes the error reduction
    for different uncertainty thresholds. The error reduction is computed as the relative error reduction
    compared to the total error. The function returns two arrays: the first array contains the relative uncertainty
    thresholds and the second array contains the corresponding relative error reductions for each threshold.
    
    :param uncertainties: An array of uncertainty values.
    :param errors: An array of true error values.
    :param error_func: A function that computes the cumulative error whose reduction is to be measured.
        default is np.mean (average error).
    :param num_bins: The number of bins to use for the uncertainty thresholds. The more bins, the more 
        fine-grained the uncertainty-error reduction curve will be
    :param percentile: The percentile margin to define the maximum and minimum uncertainty values inbetween
        which the thresholds will be computed by linear spacing according to the number of bins.
        
    :returns: two numpy arrays (uncertainty_thresholds, error_reductions) which together define the uncertainty-error
        reduction curve.
    """
    error_cum: float = error_func(errors)
    
    unc_min = np.percentile(uncertainties, percentile)
    unc_max = np.percentile(uncertainties, 100 - percentile)
    
    err_reductions = np.zeros(num_bins)
    #unc_thresholds = np.linspace(unc_min, unc_max, num_bins)
    unc_thresholds = np.linspace(unc_max, unc_min, num_bins)
    for index, th in enumerate(unc_thresholds):
        mask = uncertainties < th
        if len(errors[mask]) == 0:
            error = 0
        else:
            error = error_func(errors[mask])
        
        err_reductions[index] = (error_cum - error) / error_cum
        
    unc_thresholds = 1 - (unc_thresholds - unc_min) / (unc_max - unc_min)
    #unc_thresholds = (unc_thresholds - unc_min) / (unc_max - unc_min)
    
    return unc_thresholds, err_reductions


def negative_log_likelihood(y_true: float,
                            y_pred: float,
                            sigma: float,
                            ) -> float:
    """
    Calculates the negative log likelihood of a given true value ``y_true``, predicted value ``y_pred``
    and the predicted uncertainty ``sigma`` according to the formula:
    
    .. math::
    
        \mathrm{NLL} = 0.5 ( log(2 \pi \sigma^2) + (y_{true} - y_{pred})^2 / (sigma^2) )
    
    :param y_true: The true value.
    :param y_pred: The predicted value.
    :param sigma: The predicted uncertainty.
    
    :returns: The negative log likelihood NLL value
    """
    pred = (y_true - y_pred) ** 2 / (2 * sigma**2)
    #dist = 0.5 * np.log(np.sqrt(2 * np.pi) * sigma**2)
    dist = 0.5 * np.log(2 * np.pi * sigma**2)

    return dist + pred


def nll_score(y_true: np.ndarray, 
              y_pred: np.ndarray, 
              sigma_pred: np.ndarray
              ) -> float:
    """
    Computes the negative log likelihood (NLL) score for a given set of true values ``y_true``, predicted 
    values ``y_pred`` and the predicted uncertainties ``sigma_pred``. The negative log likelihood
    score is computed as the average of the negative log likelihoods of the Gaussian distribution
    with the predicted mean and variance for each true value. The score gives an indication of 
    how likely it is that the two sets of values (true and predicted) are drawn from the same
    distribution.
    
    :param y_true: Array of true values.
    :param y_pred: Array of predicted values.
    :param sigma_pred: Array of predicted uncertainties.
    
    :returns: The negative log likelihood score.
    """
    pred = (y_true - y_pred) ** 2 / (2 * sigma_pred**2)
    dist = 0.5 * np.log(2 * np.pi * sigma_pred**2)
    
    return np.mean(pred + dist)


def rll_score(y_true: np.ndarray,
              y_pred: np.ndarray,
              sigma_pred: np.ndarray,
              ) -> float:
    """
    Computes the *relative negative log likelihood* (RLL) score for a given set of true values ``y_true``,
    predicted values ``y_pred`` and the predicted uncertainties ``sigma_pred``.
    
    :param y_true: Array of true values.
    :param y_pred: Array of predicted values.
    :param sigma_pred: Array of predicted uncertainties.
    
    :returns: The relative negative log likelihood score.
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    nomin_values: List[float] = []
    denom_values: List[float] = []
    for y_t, y_p, sigma in zip(y_true, y_pred, sigma_pred):
        
        nomin_values.append((
            negative_log_likelihood(y_t, y_p, sigma) - 
            negative_log_likelihood(y_t, y_p, rmse)
        ))
        denom_values.append((
            negative_log_likelihood(y_t, y_p, np.abs(y_t - y_p)) -
            negative_log_likelihood(y_t, y_p, rmse)
        ))
        
    return np.sum(nomin_values) / np.sum(denom_values)  