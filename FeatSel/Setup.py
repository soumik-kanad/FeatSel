from os.path import dirname, join, exists, isdir
import argparse
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import sys
from FeatSel.Data import Data
from FeatSel.Preprocessor import Preprocessor

class Setup:
	"""
	Setup is used to set the parameters for the DatasetWrapper.

	Call setup with atleast these parameters:

		#Your data in this format
		your_data = {'data':DATA, 'target':TARGET}

		def your_evaluator_function(data, subset):
			#subset is a list of feature indexes corresponding to your_data.data.
			#compute error for the subset
			return error

		customSetup = Setup(data=your_data, evaluatorFunction=your_evaluator_function)

	Args:
		data : A dictionary containg your_data
		evaluatorFunction : A lambda function to your_evaluator_function.

	Returns:
		type: Setup object

	"""

	def __init__(self, **kwargs):
		"""
		Setup construtor.

		Kwrgs:
			data : A dictionary {'data': panda.DataFrame, 
                                'target': panda.DataFrame,
							    'feature_names': pandas.DataFrame().columns,
                                'class_name': panda.DataFrame().columns}
			evaluatorFunction : A lambda function.

		Returns:
			type: Setup object

		"""

		if kwargs != {}:
			for key, value in kwargs.items():
				if key=='data' and isinstance(value, dict):
					value = Data(value)
				setattr(self, key, value)
		else:
			self.loadDefaults()

		self.verifySetup()

	def loadDefaults(self):
		self.data_sample_folder = join(dirname(dirname(__file__)), 'datasamples')
		# self.data_sample_filename = 'featuresdf.csv'
		# self.data_sample_filename = 'elections_2013.csv'
		# self.data_sample_filename = 'Cryotherapy.xlsx'
		self.data_sample_filename = 'winequality.csv'
		self.test_size = 0.4
		self.data = Preprocessor().load(self.data_sample_folder, self.data_sample_filename, "mean").Preprocess()
		self.data = Data(self.data)
		self.f_size = self.data.data.shape[1]

		# Evaluate subset of features
		def _DefaultEvaluate(data, feature_subset):
			X = data.data[:, feature_subset]
			Y = data.target

			kf = KFold(n_splits=4)
			avg_error = 0
			for train_index, test_index in kf.split(X):
				X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
				model = LinearRegression().fit(X_train, Y_train)
				Y_pred = model.predict(X_test)
				avg_error += mean_squared_error(Y_test, Y_pred)
			
			return avg_error/4

		self.evaluatorFunction = _DefaultEvaluate

	def verifySetup(self):
		pass
