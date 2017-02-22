""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    # 1. add a column of ones in order to facilitate the array calculations with the bias
    n, m = data.shape
    n_ones = np.ones(shape=(n, 1))
    data_xo = np.concatenate((data, n_ones), axis=1)
    # 2. calculate the probability vector for class 1 probability
    z = np.dot(data_xo, weights)
    y = 1 - sigmoid(z)

    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    # 1. Cross Entropy
    n, m = y.shape
    # Mutually exclusive CE
    ce = -(np.dot(targets.T, np.log(y)))
    # Non-mutually exclusive CE
    # ce = -(np.dot(1-targets.T,np.log(1 - y)) + np.dot(targets.T,np.log(y)))

    # apply decision boundary
    for i in range(len(y)):
        if (y[i]) >= 0.5:
            y[i] = 1
        else:
            y[i] = 0

            # 2. Fraction of Inputs Classified Correctly

    incorrect = np.count_nonzero(y - targets)

    frac_correct = (n - incorrect) / float(n)

    return ce[0, 0], frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        # TODO: compute f and df without regularization
        n, m = data.shape
        n_ones = np.ones(shape=(n, 1))
        data_xo = np.concatenate((data, n_ones), axis=1)

        f = -np.dot(1.0 * targets.T, np.log(y)) - np.dot(1 - 1.0 * targets.T, np.log(1 - y))

        df = np.dot(data_xo.T, targets - (y))

    return f[0, 0], df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    alpha = hyperparameters['weight_decay']

    n, m = data.shape
    n_ones = np.ones(shape=(n, 1))
    data_xo = np.concatenate((data, n_ones), axis=1)
    z = np.dot(data_xo, weights)
    y = 1 - sigmoid(z)

    f = -np.dot(1.0 * targets.T, np.log(y)) - np.dot(1 - 1.0 * targets.T, np.log(1 - y))  - np.dot(
        np.divide(alpha, 2), np.dot(weights.T, weights))

    df = np.dot(data_xo.T, targets - (y)) - np.dot(alpha, weights)

    return f, df
