# -*- coding: utf-8 -*-
# """
# data_loader.py
#   ExampleLoader creates simple toy data for testing change-point detector.
# """

##########
#   IMPORT
##########
# 1. Built-in modules
# 2. Third-party modules
import numpy as np
# 3. Own modules


#########
#   CLASS
#########
class ExampleLoader(object):
    """
    "we have a sequence of data with change-points at t=50 and t=100.
    Data observed at each time step are generated from a single Gaussian distribution from t=1 to t=50,
    a mixture of two Gaussian distributions from t=51 to t=100,
    and a mixture of three Gaussian distributions from t = 101 to t=150.
    """
    def __init__(self, seed=None):
        if seed is None:
            pass
        else:
            np.random.seed(seed)

        simple_gaussian = np.random.normal(0, 3, size=(50, 300))

        mixture_order = np.random.choice(np.array([-1, 1]), size=(50, 300))
        double_mixture = np.random.normal(0, 1.2, size=(50, 300)) + mixture_order * 3

        mixture_order = np.random.choice(np.array([-1, 0, 1]), size=(50, 300))
        triple_mixture = np.random.normal(0, 1.0, size=(50, 300)) + mixture_order * 4.

        self.data = np.concatenate([simple_gaussian, double_mixture, triple_mixture])
        self.data_index = 0

    def sample(self):
        _sample = self.data[self.data_index]
        self.data_index = np.min([len(self.data), self.data_index + 1])

        if self.data_index == len(self.data):
            return False

        return _sample
