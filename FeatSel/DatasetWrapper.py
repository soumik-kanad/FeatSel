import pandas as pd
from FeatSel import Setup

"""
A wrapper to your dataset

This wrapper contains functions for performing feature selction and data cleaning.

Args:
    setup: A Setup object (optional)

"""

class DatasetWrapper:
	def __init__(self, setup = None):
		if setup == None:
			setup = Setup()
		assert isinstance(setup, Setup)
		for key, value in vars(setup).items():
			setattr(self, key, value)

	# Returns (error, subset of features)
	def sequential_forward(self):
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
