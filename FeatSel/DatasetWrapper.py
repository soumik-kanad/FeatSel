import pandas as pd
from FeatSel import Setup
import sys
from copy import copy
import random
import math

def exp_schedule(k=1000, lam=0.1, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * math.exp(-lam * t) if t < limit else 0)

class DatasetWrapper:
    """
    A wrapper to your dataset

    This wrapper contains functions for performing feature selction and data cleaning.

    Args:
        setup: A Setup object (optional)

    >>>data = FeatSel.DatasetWrapper() 
    >>>data.sequential_forward()
    (0.073777694751080772, [2, 4, 1])
    >>>data.simulated_annealing()
    (0.1114614410773586, [1, 4, 2])

    """

    def __init__(self, setup = None):
        if setup == None:
            setup = Setup()
        assert isinstance(setup, Setup)
        for key, value in vars(setup).items():
            setattr(self, key, value)

    # Returns (error, subset of features)
    def sequential_forward(self):
        """
        Sequential forwarding metaheuristic for subset selection.

        Returns:
            best_error : value corresponding to psuedo best fearure subset
            feature_subset : A list of feature_indexes
        """

        # All Features
        features = list(range(self.f_size))

        # Initial subset
        subset, best_error = [], float('inf')

        # Run Algorithm
        while(1):
            curr_error, curr_feat = float('inf'), None
            for feature in features:
                subset.append(feature)
                error = self.evaluatorFunction(self.data, subset)

                if error < curr_error:
                    curr_error, curr_feat = error, feature

                subset.remove(feature)

            if curr_error > best_error:
                break
            else:
                best_error = curr_error
                subset.append(curr_feat)
                features.remove(curr_feat)

        self.subset = subset
        return (best_error, subset)

    def simulated_annealing(self, schedule=exp_schedule()):
        """
        Simulated annealing metaheuristic for subset selection.

        Args :
            schedule : A lambda function for getting the time -> temperature mapping.

        Returns:
            best_error : value corresponding to psuedo best fearure subset
            feature_subset : A list of feature_indexes
        """

        def _expand(features):
            neighbors = []
            for i in range(self.f_size):
                if i in features:
                    feature_copy = copy(features)
                    feature_index = feature_copy.index(i)
                    feature_copy[feature_index:feature_index+1] = []
                else:
                    feature_copy = copy(features)
                    feature_copy.append(i)
                neighbors.append(feature_copy)
            neighbors.append([1])
            return neighbors

        def _probability(p):
            """Return true with probability p."""
            return p > random.uniform(0.0, 1.0)


        features = list(range(self.f_size))
        for t in range(sys.maxsize):
            T = schedule(t)
            if T == 0:
                return (self.evaluatorFunction(self.data, features), features)
            neighbors = _expand(features)
            if not neighbors:
                return (self.evaluatorFunction(self.data, features) ,features)
            next = random.choice(neighbors)
            if len(next) == 0:
                delta_e = sys.maxsize
            else:
                delta_e = self.evaluatorFunction(self.data, next) - self.evaluatorFunction(self.data, features)
            if delta_e < 0 or _probability(math.exp(-delta_e / T)):
                features = next
