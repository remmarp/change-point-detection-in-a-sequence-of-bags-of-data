# -*- coding: utf-8 -*-
# """
# change_detection.py
#   Simple implementation of change-detection algorithm.
# """

##########
#   IMPORT
##########
# 1. Built-in modules
import random
from collections import deque

# 2. Third-party modules
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

# 3. Own modules
from data_loader import ExampleLoader
from bayesian_bootstrap.bootstrap import mean, highest_density_interval


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


##########
#   CLASS
##########
class ChangeDetection(object):
    def __init__(self, tau=5, re_sample_trial=30, score_method='log likelihood'):
        # user parameters
        self._tau = tau  # size of reference & test window
        self._re_sample_trial = re_sample_trial  # T, which is re-sample trial number for bayesian bootstrapping
        self._score_method = score_method  # scoring method. 'log likelihood' or 'symmetrized kl'

        # reference & test window
        self._reference = deque(maxlen=tau)
        self._test = deque(maxlen=tau)

        self._xi_up = deque(maxlen=tau)  # upper bound for adaptive thresholding

        ################
        # Note:
        # For robust thresholding, I made an alarm only if cp is continuously detected in a row within a certain number.
        # ##############
        self._continuous_cp_alarm = deque(maxlen=int(np.ceil(tau/2)) + 1)

        # drawing parameters
        self._samples = []
        self._gamma = []
        self._change_point_score = []
        self._change_point_index = []  # Collect all CP points which are gamma > 0
        self._change_point_alarm_index = []  # Collect CP points which are happened continuously

    def run(self):
        data_loader = ExampleLoader()

        sample = data_loader.sample()

        step_num = 0
        scatter_time = []
        while sample is not False:
            scatter_time.append([data_loader.data_index-1] * len(sample))
            self._samples.append(sample)

            if len(self._reference) < self._reference.maxlen:
                # Wait until it collects full stack of reference window.
                self._reference.append(sample)
                self._change_point_score.append([0] * self._re_sample_trial)
                self._gamma.append(0)
            elif len(self._test) < self._test.maxlen:
                # Wait until it collects full stack of test window.
                self._test.append(sample)
                self._change_point_score.append([0] * self._re_sample_trial)
                self._gamma.append(0)
            else:
                _sample = self._test.popleft()
                self._reference.append(_sample)
                self._test.append(sample)

                _change_point_score = []
                for _idx in range(0, self._re_sample_trial):
                    _reference = define_signatures_set(self._reference)
                    _test = define_signatures_set(self._test)

                    if self._score_method == 'log likelihood':
                        _cp_score = calculate_log_likelihood_cp_score(_reference, _test)
                    elif self._score_method == 'symmetrized kl':
                        _cp_score = calculate_kl_cp_score(_reference, _test)
                    else:
                        print("Not allowed scoring method: {}.".format(self._score_method))
                        print("\tAccepted values for score method: 'log likelihood' or 'symmetrized kl")
                        break
                    _change_point_score.append(_cp_score)

                # Calculate confidence interval (standard error) for cp score mean
                _test_density = mean(_change_point_score, 10000)
                _test_low, _test_up = highest_density_interval(_test_density, alpha=0.001)

                self._change_point_score.append(_change_point_score)

                if len(self._xi_up) < self._xi_up.maxlen:
                    self._xi_up.append(_test_up)
                    self._gamma.append(0)
                else:
                    _gamma = _test_low - self._xi_up.popleft()

                    self._xi_up.append(_test_up)
                    self._gamma.append(_gamma)

                    if _gamma > 0:
                        self._change_point_index.append(step_num)
                        self._continuous_cp_alarm.append(step_num)
                        if len(self._continuous_cp_alarm) == self._continuous_cp_alarm.maxlen:
                            if self._continuous_cp_alarm[0] == step_num - self._continuous_cp_alarm.maxlen + 1:
                                if len(self._change_point_alarm_index) > 0:
                                    if self._change_point_alarm_index[-1] in self._continuous_cp_alarm:
                                        continue
                                self._change_point_alarm_index.append(self._continuous_cp_alarm[0])

            sample = data_loader.sample()
            step_num += 1

        # Drawing
        plt.figure(figsize=[15, 4])
        plt.scatter(scatter_time, self._samples, marker='o', s=10, c='black', alpha=0.1)
        plt.title('Data')
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.xlim([0, step_num+2])
        plt.xticks(np.arange(0, step_num+2, 50))
        for cp in self._change_point_alarm_index:
            if self._score_method == 'log likelihood':
                plt.axvline(x=cp - self._tau + 1, color='red', ls='--', lw=1)
            else:
                plt.axvline(x=cp - int(np.ceil(self._tau / 2)) + 1, color='red', ls='--', lw=1)
        plt.show()

        plt.figure(figsize=[15, 4])
        mu = np.mean(self._change_point_score, axis=1)
        std = np.std(self._change_point_score, axis=1)

        plt.plot(mu, ls='--', c='black', alpha=1.0)
        plt.fill_between(np.arange(0, step_num), mu - 2 * std, mu + 2 * std, color='black', alpha=0.3)
        plt.title('Change Point Score')
        plt.xlabel('Time')
        plt.ylabel('Change Point Score')
        plt.xlim([0, step_num + 2])
        plt.xticks(np.arange(0, step_num + 2, 50))
        for cp in self._change_point_alarm_index:
            plt.axvline(x=cp, color='red', ls='--', lw=1)
        plt.show()

        plt.figure(figsize=[15, 4])
        plt.plot(self._gamma, ls='--', c='black', alpha=1.0)
        plt.title('Gamma')
        plt.xlabel('Time')
        plt.ylabel('Gamma')
        plt.xlim([0, step_num + 2])
        plt.xticks(np.arange(0, step_num + 2, 50))
        plt.axhline(y=0, color='green', lw=2, ls='--')
        for cp in self._change_point_alarm_index:
            plt.axvline(x=cp, color='red', ls='--', lw=1)
        plt.show()
