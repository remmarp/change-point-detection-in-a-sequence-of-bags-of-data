# -*- coding: utf-8 -*-
# """
# util.py
#   Helper function for change detection.
# """

##########
#   IMPORT
##########
# 1. Built-in modules
import random

# 2. Third-party modules
import numpy as np
from scipy.stats import wasserstein_distance

# 3. Own modules


##############
#   DEFINITION
##############
def histogram_signatures(samples, bin_size=1):
    """
    Return signatures based on histogram. It gives bin centers and # of observation in a bin.
    :param samples: observations
    :param bin_size: size of bin
    :return:
    """
    sample = np.sort(np.reshape(samples, (-1)))  # Flatten

    min_v, max_v = np.floor(np.min(sample)), np.floor(np.max(sample))

    min_v = np.floor(min_v / bin_size) * bin_size

    bin_list = np.arange(min_v, max_v, bin_size)
    indices = 0
    signatures = []
    for _bin in bin_list:
        u = _bin + bin_size / 2.  # bin (cluster) center
        _indices = np.argmax(sample >= _bin + bin_size)
        w = _indices - indices  # number of observation in a bin
        if w == 0:
            continue
        signatures.append([u, w])
        indices = _indices

    return np.array(signatures)


def calculate_emd_score(signatures_reference, signatures_test):
    return wasserstein_distance(signatures_reference.T[0], signatures_test.T[0],
                                signatures_reference.T[1], signatures_test.T[1])


def define_signatures_set(data):
    # weight coefficients psi are re-sampled from Dirichlet distributions
    psi = np.array([1] * len(data))
    psi = np.array([random.gammavariate(_psi, 1) for _psi in psi])
    psi = np.array([_psi / sum(psi) for _psi in psi])

    signatures_set = []

    for idx in range(0, len(data)):
        sample = data[idx]
        signature = histogram_signatures(sample, bin_size=1)
        signatures_set.append([signature, psi[idx]])

    return np.array(signatures_set)


def calculate_information(reference, test):
    information = 0
    for jdx in range(0, len(test)):
        emd = calculate_emd_score(test[jdx][0], reference[0][0])
        information += test[jdx][1] * np.log(emd)

    return 0.5 + 0.5 * information


def calculate_auto_entropy(data):
    auto_entropy = 0
    for idx in range(0, len(data)):
        for jdx in range(0, len(data)):
            if idx == jdx:
                continue
            emd = calculate_emd_score(data[idx][0], data[jdx][0])
            auto_entropy += (data[idx][1] * data[jdx][1]) / (1 - data[idx][1]) * np.log(emd)

    return 0.5 + 0.5 * auto_entropy


def calculate_cross_entropy(reference, test):
    cross_entropy = 0
    for idx in range(0, len(reference)):
        for jdx in range(0, len(test)):
            emd = calculate_emd_score(reference[idx][0], test[jdx][0])
            cross_entropy += reference[idx][1] * test[jdx][1] * np.log(emd)

    return 0.5 + 0.5 * cross_entropy


def calculate_log_likelihood_cp_score(reference, test):
    s_t = np.reshape(test[0], (-1, 2))
    s_ref = reference
    s_test = test[1:]

    return calculate_information(s_t, s_ref) - calculate_information(s_t, s_test)


def calculate_kl_cp_score(reference, test):
    cross_entropy = calculate_cross_entropy(reference, test)
    auto_entropy_ref = calculate_auto_entropy(reference)
    auto_entropy_test = calculate_auto_entropy(test)

    return cross_entropy - 0.5 * (auto_entropy_ref + auto_entropy_test)
