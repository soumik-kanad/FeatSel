import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

#your_data = {'data':DATA, 'target':TARGET}
class Preprocessor(object):
	def __init__(self, type='Default'):
		self.type=type
		self.data=[];           #list of rows of data
		self.target=[];         #target value
		self.feature_names=[]   #column names
		self.class_name=""


	def load(self, path, filename):
		absolute_path = os.path.join(path, filename)
		assert os.path.exists(absolute_path)
		if filename.lower().endswith('.csv'):
			csv_load = pd.read_csv(absolute_path)
			self.feature_names = csv_load.columns[:-1]
			self.class_name = csv_load.columns[-1]
			self.data = csv_load[self.feature_names]
			self.target = csv_load[self.class_name]
		elif filename.lower().endswith('.json'):
			# TODO: Format for json type.
			pass
		elif filename.lower().endswith('.xlsx'):
			xlsx_load = pd.read_excel(absolute_path)
			self.feature_names = xlsx_load.columns[:-1]
			self.class_name = xlsx_load.columns[-1]
			self.data = xlsx_load[self.feature_names]
			self.target = xlsx_load[self.class_name]
		return self

	def Preprocess(self):
		if self.type == 'Default':
			formated_data = {}
			scaler = preprocessing.StandardScaler().fit(self.data)
			formated_data['data'] = scaler.transform(self.data)
			formated_data['target'] = self.target
			formated_data['feature_names'] = self.feature_names
			formated_data['class_name'] = self.class_name
			return formated_data
		else:
			raise NotImplementedError
