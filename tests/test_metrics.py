import numpy as np
from graph_attention_student.metrics import negative_log_likelihood, nll_score, rll_score


def test_negative_log_likelihood():
    """
    If the ``negative_log_likelihood`` function works as expected.
    """
    y_true = 1.0
    y_pred = 0.0
    sigma = 1.0
    
    nll = negative_log_likelihood(y_true, y_pred, sigma)
    
    # for these input values the NLL should be 1.4189385332046727
    assert np.isclose(nll, 1.4189385332046727)


def test_nll_score():
    """
    Tests if the ``nll_score`` function works as expected by constructing two synthetic cases:
    One in which the NLL should be good and one where it should be bad.
    """
    # Good NLL case
    y_true_good = np.array([1.0, 2.0, 3.0])
    y_pred_good = np.array([1.1, 1.8, 3.3])
    sigma_pred_good = np.array([0.1, 0.2, 0.3])
    
    nll_good = nll_score(y_true_good, y_pred_good, sigma_pred_good)
    
    # Bad NLL case
    y_true_bad = np.array([1.0, 2.0, 3.0])
    y_pred_bad = np.array([5.0, 6.2, 7.1])
    sigma_pred_bad = np.array([0.1, 0.2, 0.3])
    
    nll_bad = nll_score(y_true_bad, y_pred_bad, sigma_pred_bad)
    
    # Assert that the good NLL is lower than the bad NLL
    print('nll_good:', nll_good)
    print('nll_bad:', nll_bad)
    assert nll_good < nll_bad


def test_rll_score():
    """
    Tests if the ``rll_score`` function works as expected using the same synthetic cases as ``test_nll_score``:
    One in which the NLL should be good and one where it should be bad.
    """
    # Good NLL case
    y_true_good = np.array([1.0, 2.0, 3.0])
    y_pred_good = np.array([1.1, 1.8, 3.2])
    sigma_pred_good = np.array([0.1, 0.2, 0.3])
    
    rll_good = rll_score(y_true_good, y_pred_good, sigma_pred_good)
    
    # Bad NLL case
    y_true_bad = np.array([1.0, 2.0, 3.0])
    y_pred_bad = np.array([5.0, 6.1, 7.2])
    sigma_pred_bad = np.array([0.1, 0.2, 0.3])
    
    rll_bad = rll_score(y_true_bad, y_pred_bad, sigma_pred_bad)
    #rll_bad = rll_score(y_pred_bad, y_true_bad, sigma_pred_bad)
    
    # Assert that the good RLL is lower than the bad RLL
    print('rll_good:', rll_good)
    print('rll_bad:', rll_bad)
    assert rll_good > rll_bad
