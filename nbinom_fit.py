#!/usr/bin/env python

# fit_nbinom
# Copyright (C) 2014 Gokcen Eraslan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import numpy as np
from scipy.special import gammaln
from scipy.special import psi
from scipy.special import factorial
from scipy.optimize import fmin_l_bfgs_b as optim
from scipy.special import binom

infinitesimal = np.finfo(np.float).eps


def zero_censored_nllh(r=None, p=None, data=None):
    # We observe data that we think comes from a (r,p)
    # negative binomial distribution.  However, the data
    # is censored-- all the "zero" values are hidden
    # from us.  Compute the negative log-likelihood
    # of the censored distribution.

    if len(data) == 0:
        return(0.0)

    # Check that data has no zeroes.
    assert np.sum(data == 0) == 0

    MM = np.max(data) + 1
    nb_prob = nbinom_values(r=r, p=p, N=MM)
    censored_prob = nb_prob.copy()
    assert censored_prob[0] < 1
    censored_prob /= 1-censored_prob[0]
    neg_log_censored_prob = -np.log(censored_prob)
    censored_prob[0] = 0
    neg_log_censored_prob[0] = np.inf
    nllh = 0.0
    for d, num_d in zip(*(np.unique(data, return_counts=True))):
        nllh += num_d * neg_log_censored_prob[d]
    return(nllh)


def nbinom_values(r=None, p=None, N=None):
    # Return probability of negative binomial (r,p)
    # at k=0,1,...,N-1
    assert p < 1
    if N is None:
        # Default domain: up to 2*mean
        N = int(2*(1-p)*r/p) + 1
    domain = np.arange(N)
    values = binom(domain+r-1, r-1) * np.power(1-p, domain) * np.power(p, r)
    return(values)


def log_likelihood(params, *args):
    # Well, this is actually the negative log-likelihood
    r, p = params
    X = args[0]
    N = X.size

    # MLE estimate based on the formula on Wikipedia:
    # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
    result = np.sum(gammaln(X + r)) \
        - np.sum(np.log(factorial(X))) \
        - N * (gammaln(r)) \
        + N * r * np.log(p) \
        + np.sum(X * np.log(1 - (p if p < 1 else 1 - infinitesimal)))

    return -result

# X is a numpy array representing the data
# initial params is a numpy array representing the initial values of
# size and prob parameters


def nbinom_fit(X, initial_params=None):
    infinitesimal = np.finfo(np.float).eps

    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        # MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = np.sum(gammaln(X + r)) \
            - np.sum(np.log(factorial(X))) \
            - N * (gammaln(r)) \
            + N * r * np.log(p) \
            + np.sum(X * np.log(1 - (p if p < 1 else 1 - infinitesimal)))

        return -result

    def log_likelihood_deriv(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        pderiv = (N * r) / p - np.sum(X) / \
            (1 - (p if p < 1 else 1 - infinitesimal))
        rderiv = np.sum(psi(X + r)) \
            - N * psi(r) \
            + N * np.log(p)

        return np.array([-rderiv, -pderiv])

    if initial_params is None:
        # reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m**2) / (v - m) if v > m else 10

        # convert mu/size parameterization to prob/size
        p0 = size / ((size + m) if size + m != 0 else 1)
        r0 = size
        initial_params = np.array([r0, p0])

    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     # fprime=log_likelihood_deriv,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    params = optimres[0]
    return {'size': params[0], 'prob': params[1]}
