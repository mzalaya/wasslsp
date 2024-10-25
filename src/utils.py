# Author: Mokhtar Z. Alaya <alayaelm@utc.fr>
# License:

import numpy as np

def uniform(z):
    if abs(z) <= 1.:
        return 1.
    else:
        return 0.

def rectangle(z):
    if abs(z) <= 1.:
        return 1/2
    else:
        return 0.

def triangle(z):
    if abs(z) <= 1.:
        return (1 - abs(z))
    else:
        return 0.

def epanechnikov(z):
    if abs(z) <= 1.:
        return  3/4 * (1 - z**2)
    else:
        return 0.
def biweight(z):
    if abs(z) <= 1.:
        return 15/16 * (1 -z**2)**2
    else:
        return 0.

def tricube(z):
    if abs(z) <= 1.:
        return (1 - abs(z)**3)*3
    else:
        return 0.

def gaussian(z):
    return 1./np.sqrt(2 * np.pi) * np.exp(-z**2 / 2)

def silverman(z):
    return 1/2 * np.exp(-abs(z)/np.sqrt(2)) * np.sin(abs(z)/np.sqrt(2) + np.pi/4)

def eval_univariate(x):
    """Inner function for ECDF of one variable."""
    sorter = x.argsort()
    x = x[sorter]
    y = x.cumsum()
    y = y / y.max()
    x = np.r_[x]
    y = np.r_[y]

    return y, x

def eval_univariate_weights(x, weights):
    """Inner function for ECDF of one variable."""
    sorter = x.argsort()
    x = x[sorter]
    weights = weights[sorter]
    y = weights.cumsum()
    y = y / y.max()
    x = np.r_[x]
    y = np.r_[y]

    return y, x

